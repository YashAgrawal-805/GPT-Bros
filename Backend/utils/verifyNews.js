const { GoogleGenerativeAI } = require("@google/generative-ai");
const dotenv = require("dotenv");

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.GOO_API);
const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

async function verifyNews(newsText) {
  try {
    // ensure it's always a string
    const cleanText = typeof newsText === "string" 
      ? newsText 
      : JSON.stringify(newsText);

    const prompt = `choose one news among all
    News Article: "${cleanText}"`;

    const response = await model.generateContent(prompt);
    return response.response.text();
  } catch (error) {
    console.error("Error verifying news:", error);
    throw new Error("Failed to verify news");
  }
}

module.exports = verifyNews;
