from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from rag2 import retrieve
from tool import note_duration_ms

load_dotenv()
api_key = os.environ.get("ANTHROPIC_API_KEY")

llm = ChatAnthropic(model="claude-sonnet-5", api_key=api_key, max_tokens=200)
llm_with_tools = llm.bind_tools([note_duration_ms])


def answer_question(question):
    try:
        retrieved_text = retrieve(question)

        if retrieved_text:
            prompt = f"""Answer the question using only this context:
    
Context:{retrieved_text}
    
Question:{question}"""
        else:
            prompt = f"""No relevant information was found in the knowledge base for this question.
If it can be answered using the note_duration_ms tool, use it. Otherwise, say you don't have information on this topic.

Question:{question}"""

        messages = [{"role":"user", "content": prompt}]
        ai_response = llm_with_tools.invoke(messages)

        if not ai_response.tool_calls:
            return ai_response.content

        messages.append(ai_response)
        for tool_call in ai_response.tool_calls:
            if tool_call["name"] == "note_duration_ms":
                result = note_duration_ms.invoke(tool_call["args"])
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call["id"]
                })

        final_response = llm_with_tools.invoke(messages)
        return final_response.content

    except Exception as e:
        print(f"Agent error: {e}")
        return "Sorry, something went wrong. Please try again."

if __name__ == "__main__":
    print(answer_question("What instrument did Adolphe Sax invent?"))
    print(answer_question("How long is a quarter note at 100 BPM?"))
    print(answer_question("What is the capital of Japan?"))

