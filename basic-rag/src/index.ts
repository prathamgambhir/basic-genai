import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import readlineSync from "readline-sync";
import { tool } from "@langchain/core/tools";
import z from "zod";
import { createAgent, SystemMessage } from "langchain";

//pdf loading
const pdfPath = "../dsa_c.pdf";
const loader = new PDFLoader(pdfPath);
const rawDocs = await loader.load();

//creating pdf chunks
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200
})

const finalDoc = await textSplitter.splitDocuments(rawDocs);

//initializing Embedding model
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GOOGLE_GENAI_API_KEY as string,
    // model: "gemini-embedding-001"
})

//initialize pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY as string,
});
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME as string);

const vectorStore = new PineconeStore(embeddings, {
  pineconeIndex,
  // maxConcurrency: 5,
});

const retrieveSchema = z.object({ query: z.string() });

const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);

const tools = [retrieve];

const systemPrompt = new SystemMessage(
    "You have access to a tool that retrieves context from a blog post. " +
    "Use the tool to help answer user queries."
)

const agent = createAgent({ model: "gemini-2.5-flash", tools, systemPrompt });


async function main() {
    const userProblem = readlineSync.question("Ask me anything: ");
    
}