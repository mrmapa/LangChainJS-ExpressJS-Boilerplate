import { MessagesPlaceholder, PromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { TaskType } from "@google/generative-ai";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

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

      console.log(res);

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
      // Load and chunk contents of blog
      const pTagSelector = "p";
      const cheerioLoader = new CheerioWebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        {
          selector: pTagSelector
        }
      );

      const docs = await cheerioLoader.load();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000, chunkOverlap: 200
      });
      const allSplits = await splitter.splitDocuments(docs);


      // Index chunks
      await vectorStore.addDocuments(allSplits)

      // Define prompt for question-answering
      const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

      // Define state for application
      const InputStateAnnotation = Annotation.Root({
        question: Annotation<string>,
      });

      const StateAnnotation = Annotation.Root({
        question: Annotation<string>,
        context: Annotation<Document[]>,
        answer: Annotation<string>,
      });

      // Define application steps
      const retrieve = async (state: typeof InputStateAnnotation.State) => {
        const retrievedDocs = await vectorStore.similaritySearch(state.question)
        return { context: retrievedDocs };
      };


      const generate = async (state: typeof StateAnnotation.State) => {
        const docsContent = state.context.map(doc => doc.pageContent).join("\n");
        const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
        const response = await llm.invoke(messages);
        return { answer: response.content };
      };


      // Compile application and test
      const graph = new StateGraph(StateAnnotation)
        .addNode("retrieve", retrieve)
        .addNode("generate", generate)
        .addEdge("__start__", "retrieve")
        .addEdge("retrieve", "generate")
        .addEdge("generate", "__end__")
        .compile();

        let inputs = { question: "What is Task Decomposition?" };

        const result = await graph.invoke(inputs);
        console.log(result.answer);
    },
  }
]
