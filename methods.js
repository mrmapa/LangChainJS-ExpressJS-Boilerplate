import { MessagesPlaceholder, PromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TaskType } from "@google/generative-ai";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { toolsCondition } from "@langchain/langgraph/prebuilt";
import { isAIMessage } from "@langchain/core/messages";
import * as fs from 'node:fs';
import * as readline from 'node:readline';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const prettyPrint = (message) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};
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

      const date = new Date();

      const day = date.getDate();
      const month = date.getMonth() + 1; // The month index starts from 0
      const year = date.getFullYear();

      const prompt = ChatPromptTemplate.fromMessages([
              [
                "system",
               `You will be functioning as a chatbot called Connexx Bot designed for helping a user become more active.
               User's full name is ${input['Name']}. Always address the user by using their first name. User's age is ${input['Age']}.
               Always reference user's age when it makes sense in your answer. User's height is ${input['Height_Feet']} feet, ${input['Height_Inches']}inches.
               Always reference user's height when it makes sense in your answer. User's weight is ${input['Weight']}.
               Always reference user's weight if it makes sense in your answer.
               Always keep the subject around fitness and subtopics around fitness.
               If subject is not under this scope respond with "Sorry, I can't help with that."
               Offer to create a workout schedule if it makes sense in your answer. If the user requests a schedule, response should include "Here is your proposed schedule:".
               Before generating a schedule, ask clarifying questions to determine what equipment the user has available (e.g. full gym, nothing), what their experience is, and
               when they have time and how much time they have to perform the workouts.
               Make sure any schedule given repeats at a regular interval. Events in the generated schedule should include the start and end time of the workout.
               Make sure that the schedule starts after the current date. The current month is ${month}, the current day is ${day}, and the current year is ${year}.
               After generating a first draft of the schedule, ask the user if there is anything they would like modified. Make sure that the user is ok with the times, days of the week, and
               exercises in the schedule, and if they are not, modify the schedule until the user indicates that they are happy with the proposed schedule.
               After the user is happy with the schedule, ask the user if they would like you to generate a link that can be added to their Google Calendar. If they say yes,
               ask the user how many weeks they would like the schedule to go for, then generate a single google calendar invite. Modify the timestamps to match the timezone
               of the user.
               This should always be at the end of the response and formatted as a single JSON snippet:
               {{
                "title": "Name of event",
                "description": "Description",
                "startTime": "YYYYMMDDTHHMMSS",
                "endTime": "YYYYMMDDTHHMMSS",
                "location": "Location",
                "recurrence": {{
                  "frequency": "WEEKLY",
                  "days": ["MO", "WE", "FR"],
                  "until": "YYYYMMDDTHHMMSS"
                }}
              }}
                
              After generating a schedule, ask the user if there is anything else you could help them with.`
              ],
              new MessagesPlaceholder("msgs"),
              ["human", input['Input']],
            ]);

      const chain = prompt.pipe(chat);
      const res = await chain.invoke({msgs: input['msgs']});

      return res
    },
  },
  {
    id: "call-model-rag",
    route: "/call-model-rag",
    method: "post",
    description:
      "Calls the Gemini API.",
    inputVariables: ["Input", "Message_History", "Name", "Age", "Height_Feet", "Height_Inches", "Weight"],
    execute: async (input) => {
      const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0,
        maxRetries: 2,
        apiKey: process.env.GEMINI_API_KEY
      });

      const embeddings = new GoogleGenerativeAIEmbeddings({
        model: "text-embedding-004", // 768 dimensions
        taskType: TaskType.RETRIEVAL_DOCUMENT,
        title: "Document title",
        apiKey: process.env.GEMINI_API_KEY
      });

      const vectorStore = new MemoryVectorStore(embeddings);

      // read in web documents
      const urls = ["https://www.runnersworld.com/uk/training/beginners/a30279918/run-walk-plan-complete-beginners/",
        "https://run.outsideonline.com/training/getting-started/how-to-start-running-training-plan/"];

      // load each web doc
      const pTagSelector = "p";
      let docs = [];
      for (let i = 0; i < urls.length; i++) {
        const cheerioLoader = new CheerioWebBaseLoader(
          urls[i],
          {
            selector: pTagSelector
          }
        );
        const new_doc = await cheerioLoader.load();
        if (docs.length == 0) {
          docs = new_doc;
        }
        else {
          docs = docs.concat([new_doc]);
        }
      }
      console.log(docs);

      // Load and chunk contents of pdf
      const pdfPath = "docs/pdf/Pages-from-APES_1_2016_za_mail-31.pdf";

      const pdfLoader = new PDFLoader(pdfPath);

      const pdfDocs = await pdfLoader.load();
      console.log(pdfDocs.length);

      docs = docs.concat(pdfDocs);
      console.log(docs.length);
      console.log(docs);
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, chunkOverlap: 200
      });
      const allSplits = await splitter.splitDocuments(docs);

      // Index chunks
      await vectorStore.addDocuments(allSplits);

      const retrieveSchema = z.object({ query: z.string() });

      const retrieve = tool(
        async ({ query }) => {
          const retrievedDocs = await vectorStore.similaritySearch(query, 5);
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

      // Step 1: Generate an AIMessage that may include a tool-call to be sent.
      async function queryOrRespond(state) {
        const llmWithTools = llm.bindTools([retrieve]);
        const response = await llmWithTools.invoke(state.messages);

        // MessagesState appends messages to state instead of overwriting
        return { messages: [response] };
      }

      // Step 2: Execute the retrieval.
      const tools = new ToolNode([retrieve]);

      // Step 3: Generate a response using the retrieved content.
      async function generate(state) {
        // Get generated ToolMessages
        let recentToolMessages = [];
        for (let i = state["messages"].length - 1; i >= 0; i--) {
          let message = state["messages"][i];
          if (message instanceof ToolMessage) {
            recentToolMessages.push(message);
          } else {
            break;
          }
        }
        let toolMessages = recentToolMessages.reverse();

        // Format into prompt
        const date = new Date();

        const day = date.getDate();
        const month = date.getMonth() + 1; // The month index starts from 0
        const year = date.getFullYear();

        const docsContent = toolMessages.map((doc) => doc.content).join("\n");
        console.log(docsContent);
        const systemMessageContent =
            `You will be functioning as a chatbot called Connexx Bot designed for helping a user become more active.
               User's full name is ${input['Name']}. Always address the user by using their first name. User's age is ${input['Age']}.
               Always reference user's age when it makes sense in your answer. User's height is ${input['Height_Feet']} feet, ${input['Height_Inches']}inches.
               Always reference user's height when it makes sense in your answer. User's weight is ${input['Weight']}.
               Always reference user's weight if it makes sense in your answer.
               Always keep the subject around fitness and subtopics around fitness.
               If subject is not under this scope respond with "Sorry, I can't help with that."
               Offer to create a workout schedule if it makes sense in your answer. If the user requests a schedule, response should include "Here is your proposed schedule:".
               Before generating a schedule, ask clarifying questions to determine what equipment the user has available (e.g. full gym, nothing), what their experience is, and
               when they have time and how much time they have to perform the workouts.
               Make sure any schedule given repeats at a regular interval. Events in the generated schedule should include the start and end time of the workout.
               Make sure that the schedule starts after the current date. The current month is ${month}, the current day is ${day}, and the current year is ${year}.
               After generating a first draft of the schedule, ask the user if there is anything they would like modified. Make sure that the user is ok with the times, days of the week, and
               exercises in the schedule, and if they are not, modify the schedule until the user indicates that they are happy with the proposed schedule.
               After the user is happy with the schedule, ask the user if they would like you to generate a link that can be added to their Google Calendar. If they say yes,
               ask the user how many weeks they would like the schedule to go for, then generate a single google calendar invite. Modify the timestamps to match the timezone
               of the user.
               This should always be at the end of the response and formatted as a single JSON snippet:
               {{
                "title": "Name of event",
                "description": "Description",
                "startTime": "YYYYMMDDTHHMMSS",
                "endTime": "YYYYMMDDTHHMMSS",
                "location": "Location",
                "recurrence": {{
                  "frequency": "WEEKLY",
                  "days": ["MO", "WE", "FR"],
                  "until": "YYYYMMDDTHHMMSS"
                }}
              }}
                
              After generating a schedule, ask the user if there is anything else you could help them with.
              Use the following pieces of retrieved context to help answer the question.` +
              "\n\n" +
              `${docsContent}`;

        const conversationMessages = state.messages.filter(
          (message) =>
            true
        );

        const prompt = [
          new SystemMessage(systemMessageContent),
          ...conversationMessages,
        ];

        // Run
        const response = await llm.invoke(prompt);
        return { messages: [response] };
      }

      const graphBuilder = new StateGraph(MessagesAnnotation)
      .addNode("queryOrRespond", queryOrRespond)
      .addNode("tools", tools)
      .addNode("generate", generate)
      .addEdge("__start__", "queryOrRespond")
      .addConditionalEdges("queryOrRespond", toolsCondition, {
        __end__: "__end__",
        tools: "tools",
      })
      .addEdge("tools", "generate")
      .addEdge("generate", "__end__");

      const graph = graphBuilder.compile();

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

      let inputs = {
        messages: input['msgs'],
      };

      let output;
      
      for await (const step of await graph.stream(inputs, {
        streamMode: "values",
      })) {
        const lastMessage = step.messages[step.messages.length - 1];
        prettyPrint(lastMessage);
        console.log("-----\n");
        output = lastMessage;
      }
      
      return output;
    },
  }
]
