import dotenv from "dotenv";
import OpenAI from "openai";
import cosineSimilarity from "compute-cosine-similarity";
import readline from "node:readline";

dotenv.config();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Canonical reference activity types
const activities = [
	"Inventory Check",
	"Customer Support",
	"Staff Meeting",
	"Cleaning Duty",
	"Cash Register Operation",
];

const HIGH_CONFIDENCE_THRESHOLD = 0.92;
const REVIEW_THRESHOLD = 0.8;

async function getEmbedding(text) {
	const res = await openai.embeddings.create({
		model: "text-embedding-3-small",
		input: text,
	});
	return res.data[0].embedding;
}

async function classifyWithLLM(input, activities) {
	const prompt = `
You are a helpful assistant for classifying employee shift activities.
Choose the most appropriate standardized activity from this list:

${activities.map((a, i) => `${i + 1}. ${a}`).join("\n")}

Description: "${input}"

Respond in JSON format:
{
  "match": "<activity name>",
  "confidence": <value between 0 and 1>
}
`;

	const res = await openai.chat.completions.create({
		model: "gpt-4",
		messages: [{ role: "user", content: prompt }],
		temperature: 0.2,
	});

	try {
		return JSON.parse(res.choices[0].message.content);
	} catch (err) {
		throw new Error(
			`Failed to parse LLM response: ${res.choices[0].message.content}`
		);
	}
}

async function mapEntry(inputEntry) {
	const inputEmbedding = await getEmbedding(inputEntry);

	let bestMatch = null;
	let bestScore = -1;

	for (const activity of activities) {
		const activityEmbedding = await getEmbedding(activity);
		const score = cosineSimilarity(inputEmbedding, activityEmbedding);
		if (score > bestScore) {
			bestMatch = activity;
			bestScore = score;
		}
	}

	console.log(
		`Embedding match: "${bestMatch}" with score ${bestScore.toFixed(3)}`
	);

	if (bestScore >= HIGH_CONFIDENCE_THRESHOLD) {
		return { type: bestMatch, source: "embedding", confidence: bestScore };
	}

	if (bestScore >= REVIEW_THRESHOLD) {
		const llmResult = await classifyWithLLM(inputEntry, activities);
		console.log(
			`LLM match: "${llmResult.match}" with confidence ${llmResult.confidence}`
		);
		if (llmResult.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
			return {
				type: llmResult.match,
				source: "llm",
				confidence: llmResult.confidence,
			};
		} else {
			return { type: null, source: "review", confidence: llmResult.confidence };
		}
	}

	return { type: null, source: "review", confidence: bestScore };
}

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});

rl.question("Enter unmapped activity: ", async (input) => {
	console.log(`You entered: ${input}`);

	const result = await mapEntry(input);

	if (result.type) {
		console.log(
			`✅ Mapped to: ${result.type} (via ${
				result.source
			}, confidence: ${result.confidence.toFixed(2)})`
		);
	} else {
		console.log(
			`⚠️ Sent for review (confidence: ${result.confidence.toFixed(2)})`
		);
	}

	rl.close();
});
