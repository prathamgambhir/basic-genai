import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import readlineSync from "readline-sync";
import { tool } from "@langchain/core/tools";
import z from "zod";
import { createAgent, HumanMessage, AIMessage } from "langchain";

const pdfPath = "./c_dsa.pdf";
const loader = new PDFLoader(pdfPath);
const rawDocs = await loader.load();
console.log(`✅ Loaded ${rawDocs.length} pages`);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const finalDoc = (await textSplitter.splitDocuments(rawDocs)).filter(
  (doc) => doc.pageContent && doc.pageContent.trim().length > 0,
);
console.log(finalDoc[0]);
console.log(`✅ Created ${finalDoc.length} chunks`);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GOOGLE_GEMINI_API_KEY as string,
  model: "text-embedding-001",
});

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME as string);
const vectorStore = new PineconeStore(embeddings, { pineconeIndex , maxConcurrency: 5});
// let vectorStore: PineconeStore;

console.log("📝 Storing embeddings...");

// await PineconeStore.fromDocuments(finalDoc, embeddings, {
//     pineconeIndex,
//     maxConcurrency: 5,
// });
// console.log("✅ Embeddings stored successfully");

try {
  //   if (finalDoc.length > 0) {
  //     // Use addDocuments; if it still errors, the index dimension might be wrong (should be 768)
  //     vectorStore = await PineconeStore.fromDocuments(finalDoc, embeddings, {
  //       pineconeIndex,
  //       maxConcurrency: 5,
  //     });
  //     console.log("✅ Embeddings stored");
  //   }
  // if (finalDoc.length > 0) {
  //     // We use manual batching to avoid the "Must pass at least 1 record" error
  //     // This ensures the SDK always has a valid batch to send
  //     const batchSize = 50;
  //     for (let i = 0; i < finalDoc.length; i += batchSize) {
  //         const batch = finalDoc.slice(i, i + batchSize);
  //         await vectorStore.addDocuments(batch);
  //         console.log(`   Uploaded chunks ${i} to ${Math.min(i + batchSize, finalDoc.length)}`);
  //     }
  //     console.log("✅ Embeddings stored successfully");
  // }
  // if (finalDoc.length === 0) {
  //   console.error("❌ No valid chunks to store");
  //   process.exit(1);
  // }

  // console.log("📝 Storing embeddings...");

  // vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  //   pineconeIndex,
  // });

  // const batchSize = 50;

  // for (let i = 0; i < finalDoc.length; i += batchSize) {
  //   const batch = finalDoc.slice(i, i + batchSize);

  //   // EXTRA SAFETY: filter again
  //   const safeBatch = batch.filter(
  //     (d) => d.pageContent && d.pageContent.trim().length > 0,
  //   );

  //   if (safeBatch.length === 0) continue;

  //   await vectorStore.addDocuments(safeBatch);

  //   console.log(
  //     `✅ Uploaded ${i} → ${Math.min(i + batchSize, finalDoc.length)}`,
  //   );
  // }

  await vectorStore.addDocuments(finalDoc);
  console.log("🎉 All embeddings stored successfully");
} catch (error) {
  console.log(
    "⚠️ Storage note: ",
    error instanceof Error ? error.message : error,
  );
  process.exit(1);
}

const retrieveSchema = z.object({ query: z.string() });
const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`,
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve info about DSA in C from the PDF.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  },
);

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash", // Recommended stable version
  apiKey: process.env.GOOGLE_GEMINI_API_KEY as string,
});

const agent = createAgent({
  model,
  tools: [retrieve],
  systemPrompt:
    "You are a helpful assistant. Use the 'retrieve' tool to answer questions about DSA in C from the provided documents.",
});

async function main() {
  console.log("\n🤖 RAG Application Ready! (Type 'exit' to quit)\n");
  const conversationHistory: { type: string; content: string }[] = [];

  while (true) {
    const userProblem = readlineSync.question("Ask me anything: ");

    if (["exit", "quit"].includes(userProblem.toLowerCase())) break;
    if (!userProblem.trim()) continue;

    try {
      console.log("🔍 Thinking...");

      const messages: any[] = [];
      // DO NOT push a SystemMessage here. The agent already has it from the constructor.

      for (const msg of conversationHistory) {
        if (msg.type === "human") messages.push(new HumanMessage(msg.content));
        else if (msg.type === "ai") messages.push(new AIMessage(msg.content));
      }

      messages.push(new HumanMessage(userProblem));

      const response = await agent.invoke({ messages });
      const answer = response?.messages[response.messages.length - 1]?.content;

      conversationHistory.push({ type: "human", content: userProblem });
      if (answer)
        conversationHistory.push({ type: "ai", content: answer.toString() });

      console.log(
        `\n📝 Answer:\n${"-".repeat(40)}\n${answer}\n${"-".repeat(40)}\n`,
      );
    } catch (error) {
      console.log("\n❌ Error:", error);
    }
  }
}

main().catch(console.error);
