import dotenv from "dotenv";
import OpenAI from "openai";
import cosineSimilarity from "compute-cosine-similarity";
import { readFile, writeFile } from "fs/promises";

const INPUT_FILE_PATH = "./src/data/input.json";
const OUTPUT_FILE_PATH = "./src/data/output.json";
const CANONICAL_ACTIVITIES = [
	"absence, planned",
	"absence, unplanned",
	"customer facing time, phone inbound",
	"customer facing time, phone outbound",
	"customer facing time, web chat",
	"customer facing time, social media",
	"holiday, holiday",
	"holiday, bank holiday",
	"holiday, emergency",
	"planned event, callback",
	"planned event, coaching",
	"planned event, meeting",
];

const HIGH_CONFIDENCE_THRESHOLD = 0.9;
const REVIEW_THRESHOLD = 0.7;

dotenv.config();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

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

	for (const activity of CANONICAL_ACTIVITIES) {
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

	const mappingResult = {
		input: inputEntry,
		mappedOutput: bestMatch,
		source: "embedding",
		confidence: bestScore,
		humanReviewRequired: true,
	};

	if (bestScore >= HIGH_CONFIDENCE_THRESHOLD) {
		console.log(
			"score is above high confidence threshold, no LLM review required"
		);

		mappingResult.humanReviewRequired = false;

		return mappingResult;
	}

	if (bestScore >= REVIEW_THRESHOLD) {
		console.log(
			"score is below high confidence threshold but above review threshold, using LLM for classification"
		);

		const llmResult = await classifyWithLLM(inputEntry, CANONICAL_ACTIVITIES);

		console.log(
			`LLM match: "${llmResult.match}" with confidence ${llmResult.confidence}`
		);

		if (llmResult.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
			console.log(
				"LLM review score is above high confidence threshold, no human review required"
			);

			mappingResult.mappedOutput = llmResult.match;
			mappingResult.source = "llm";
			mappingResult.confidence = llmResult.confidence;
			mappingResult.humanReviewRequired = false;

			return mappingResult;
		} else {
			console.log(
				"LLM review score is below high confidence threshold, human review required"
			);

			mappingResult.source = "llm";
			mappingResult.humanReviewRequired = true;

			if (llmResult.confidence >= mappingResult.confidence) {
				mappingResult.mappedOutput = llmResult.match;
				mappingResult.confidence = llmResult.confidence;
			}

			return mappingResult;
		}
	}

	return mappingResult;
}

async function processEntry(input) {
	console.log(`Processing entry: "${input}"`);

	const result = await mapEntry(input);

	if (result.type) {
		console.log(
			`"${input}" mapped to: ${
				result.type
			} (confidence: ${result.confidence.toFixed(2)})`
		);
	} else {
		console.log(
			`"${input}" mapped to: ${
				result.type
			} and flagged for review (confidence: ${result.confidence.toFixed(2)})`
		);
	}

	return result;
}

async function processInputFile() {
	console.log("Processing input file...");

	const rawData = await readFile(INPUT_FILE_PATH, "utf-8");
	const jsonData = JSON.parse(rawData);

	const mappedData = await Promise.all(
		jsonData.map(async (item) => {
			return await processEntry(item);
		})
	);

	console.log("Mapping complete. Writing results to output file...");

	await writeFile(OUTPUT_FILE_PATH, JSON.stringify(mappedData, null, 2));
}

await processInputFile();
