import { GoogleGenAI, Type } from "@google/genai";
import "dotenv/config";
import readlineSync from "readline-sync";

// Configure the client
const ai = new GoogleGenAI({});

async function getBitcoin({ coin }) {
  const response = await fetch(
    `https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr&ids=${coin}`,
  );
  const data = await response.json();
  console.log(data);
  return data;
}

async function getStockPrice({ ticker }) {
  const response = await fetch(
    `https://api.api-ninjas.com/v1/stockprice?ticker=${ticker}`,
    { headers: { "X-Api-Key": process.env.NINJA_API_KEY } },
  );
  const data = await response.json();
  console.log(data);
  return data;
}

async function getWeather({ location }) {
  const response = await fetch(
    `http://api.weatherapi.com/v1/current.json?key=d6a3bcd7a43c4ed59c2155208252404&q=${location}&aqi=no`,
  );
  const data = await response.json();
  console.log(data);
  return data;
}

const getBitcoinDeclaration = {
  name: "getBitcoin",
  description: "Get the current price of the given cryptcurrency.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      coin: {
        type: Type.STRING,
        description:
          "The name of cryptocurrency e.g. bitcoin, ethereum, dogecoin etc.",
      },
    },
    required: ["coin"],
  },
};

const getStockPriceDeclaration = {
  name: "getStockPrice",
  description: "Get the current price of the given stock.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      ticker: {
        type: Type.STRING,
        description:
          "The ticker symbol of the stock. e.g. MSFT, AAPL, GOOGL etc.",
      },
    },
    required: ["ticker"],
  },
};

const getWeatherDeclaration = {
  name: "getWeather",
  description: "Get the weather for the given location.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      location: {
        type: Type.STRING,
        description:
          "The name of the location. e.g. New York, Mumbai, Delhi etc.",
      },
    },
    required: ["location"],
  },
};

const History = [];

const tools = [
  {
    functionDeclarations: [
      getBitcoinDeclaration,
      getStockPriceDeclaration,
      getWeatherDeclaration,
    ],
  },
];

const toolFunctions = {
  getBitcoin: getBitcoin,
  getStockPrice: getStockPrice,
  getWeather: getWeather,
};

async function runAgent() {
  while (true) {
    const agentResponse = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: History,
      config: { tools },
    });

    if (agentResponse.functionCalls && agentResponse.functionCalls.length > 0) {
      console.log("Function called");
      const functionCall = agentResponse.functionCalls[0];

      const { name, args } = functionCall;

      History.push({
        role: "model",
        parts: [{ functionCall: functionCall }],
      });

      const funcResponse = await toolFunctions[name](args);

      const functionResponsePart = {
        name: functionCall.name,
        response: { result: funcResponse },
      };

      History.push({
        role: "user",
        parts: [{ functionResponse: functionResponsePart }],
      });
    } else {
      History.push({
        role: "model",
        parts: [{ text: agentResponse.text }],
      });

      console.log(agentResponse.text);
      break;
    }
  }
}

while (true) {
  const question = readlineSync.question("Ask me anything: ");

  if (question === "exit") {
    break;
  }

  History.push({
    role: "user",
    parts: [{ text: question }],
  });

  await runAgent();
}
