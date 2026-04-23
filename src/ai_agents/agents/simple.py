import os
import subprocess
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from ai_agents.config.settings import settings



#################################################
##################### Tools #####################
#################################################
@tool
def write_python_script(file_path: str, code: str) -> str:
    """Writes Python code to a specified file path."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        return f"Success: Script written to {file_path}"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"



@tool
def execute_python_script(file_path: str) -> str:
    """Executes a Python script locally and returns the standard output or error."""
    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            timeout=10 
        )

        if result.returncode == 0:
            return f"Execution successful.\nOutput:\n{result.stdout}"
        else:
            return f"Execution failed.\nError:\n{result.stderr}"
        
    except Exception as e:
        return f"Failed to run script: {str(e)}"





#################################################
##################### Agent #####################
#################################################
def initialize_coding_agent():

    llm = ChatGroq(
        model=settings.chat_model, 
        api_key=settings.resolved_groq_api_key(),
        temperature=0.0 # Strict determinism
    )
    
    tools = [write_python_script, execute_python_script]
    
    # The system prompt dictates the agent's behavior and lifecycle
    system_prompt = (
        "You are an autonomous coding agent. Your workflow is as follows: "
        "1. Write clean, documented Python code. "
        "2. Save the code to the file system using the write_python_script tool. "
        "3. Execute the script using the execute_python_script tool to verify it works. "
        "4. If execution fails, analyze the error output and repeat the process to fix it."
    )
    

    agent = create_agent(
        llm, 
        tools, 
        system_prompt=system_prompt
    )
    
    return agent






if __name__ == "__main__":
    print("Groq API Key:", settings.resolved_groq_api_key())
    print("\n\n\n\n")
    agent = initialize_coding_agent()
    
    # A simple task for the agent to complete
    task = "Write a python script at 'workspace/math_ops.py' that calculates the Fibonacci sequence up to 10. Then run it."
    
    print(f"Assigning Task: {task}\n")
    print("-" * 50)
    
    # Stream the graph state to watch the agent reason, act, and observe
    for chunk in agent.stream({"messages": [("human", task)]}):
        if "agent" in chunk:
            print(f"Agent Action: {chunk['agent']['messages'][-1].content}\n")
            
        elif "tools" in chunk:
            print(f"Tool Output: {chunk['tools']['messages'][-1].content}\n")