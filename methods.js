import { OpenAI } from "langchain/llms"
import {
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts"
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { LLMChain } from "langchain/chains"
import { PassThrough } from "stream"
import { CallbackManager } from "langchain/callbacks"

export const methods = [
  {
    id: "call-model",
    route: "/call-model",
    method: "post",
    description:
      "Calls the Gemini API.",
    inputVariables: ["Input", "Message_History", "Name", "Age", "Height_Feet", "Height_Inches", "Weight"],
    execute: async (input) => {
      const chat = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        maxRetries: 2,
        apiKey: process.env.GEMINI_API_KEY
      });

      // parse message history
      const history = input['Message_History'].split("|");
      let messages = [];

      history.slice(0, -1).forEach((element) => {
        let role_msg = element.split('^');
        if (role_msg[0] == 'user') {
          let new_msg = new HumanMessage(role_msg[1]);
          messages.push(new_msg);
        }
        else {
          let new_msg = new AIMessage(role_msg[1]);
          messages.push(new_msg);
        }
      });

      input['msgs'] = messages;

      const prompt = ChatPromptTemplate.fromMessages([
              [
                "system",
               `You will be functioning as a chatbot called Connexx Bot designed for helping a user become more active.
               User's full name is {Name}. Always address the user by using their first name. User's age is {Age}.
               Always reference user's age when it makes sense in your answer. User's height is {Height_Feet} feet, {Height_Inches} inches.
               Always reference user's height when it makes sense in your answer. User's weight is {Weight}.
               Always reference user's weight if it makes sense in your answer.
               Always keep the subject around fitness and subtopics around fitness.
               If subject is not under this scope respond with "Sorry, I can't help with that."`,
              ],
              new MessagesPlaceholder("msgs"),
              ["human", "{Input}"],
            ]);
      const chain = new LLMChain({ llm: chat, prompt: prompt })
      const res = await chain.call(input);

      return res
    },
  }
]
