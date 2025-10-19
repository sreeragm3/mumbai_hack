import os
import getpass
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Import our custom ML tools
from legacy.ml_tools import predict_house_price, get_model_info

# Load environment variables
load_dotenv()

class HousePriceAgent:
    def __init__(self):
        """Initialize the LangChain agent with Gemini and ML tools"""
        self.setup_api_key()
        self.setup_model()
        self.setup_tools()
        self.setup_agent()
    
    def setup_api_key(self):
        """Setup Google API key"""
        if not os.environ.get("GOOGLE_API_KEY"):
            api_key = getpass.getpass("Enter your Google Gemini API key: ")
            os.environ["GOOGLE_API_KEY"] = api_key
    
    def setup_model(self):
        """Initialize Gemini model with updated model name"""
        try:
            # Updated model names that are currently available
            model_options = [
                "gemini-2.0-flash",       # Latest stable model (recommended)
                "gemini-2.5-flash",       # Alternative option
                "gemini-1.5-flash-002",   # Stable version
                "gemini-1.5-pro-002"     # Pro version
            ]
            
            # Try models in order of preference
            for model_name in model_options:
                try:
                    self.model = init_chat_model(
                        model_name, 
                        model_provider="google_genai",
                        temperature=0.3
                    )
                    print(f"✅ Successfully initialized {model_name}")
                    break
                except Exception as e:
                    print(f"⚠️ Failed to initialize {model_name}: {e}")
                    if model_name == model_options[-1]:  # Last option
                        raise e
                    continue
                    
        except Exception as e:
            print(f"❌ Error initializing any Gemini model: {e}")
            print("\n🔧 Troubleshooting tips:")
            print("1. Make sure your Google API key is valid")
            print("2. Check that the Gemini API is enabled in your Google Cloud project")
            print("3. Try updating the langchain-google-genai package: pip install -U langchain-google-genai")
            raise
    
    def setup_tools(self):
        """Setup available tools for the agent"""
        self.tools = [
            predict_house_price,
            get_model_info
        ]
        print(f"✅ Loaded {len(self.tools)} tools")
    
    def setup_agent(self):
        """Create the agent with tools and prompt"""
        # Create system prompt
        system_prompt = """
        You are an intelligent real estate AI assistant that can help users predict house prices 
        using machine learning. You have access to a trained ML model that can predict whether 
        a house will be in Low, Medium, or High price categories.

        Your capabilities:
        1. Predict house price categories based on property features
        2. Explain predictions and provide insights
        3. Help users understand what factors influence house prices
        4. Provide information about the ML model being used

        When users ask about house price predictions:
        1. First, gather the required information (bedrooms, bathrooms, square_feet, age)
        2. Use the prediction tool to get the ML model's prediction
        3. Explain the result in a user-friendly way
        4. Provide insights about factors that might influence the prediction

        Be helpful, accurate, and explain technical concepts in simple terms.
        """
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(
            self.model, 
            self.tools, 
            prompt
        )
        
        # Create executor
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools,
            verbose=True,
            max_iterations=3
        )
        
        print("✅ Agent created successfully")
    
    def chat(self, message: str) -> str:
        """Process user message and return agent response"""
        try:
            response = self.agent_executor.invoke({
                "input": message,
                "chat_history": []
            })
            return response["output"]
        except Exception as e:
            return f"❌ Error processing message: {str(e)}"
    
    def run_interactive_session(self):
        """Run an interactive chat session"""
        print("\n🏠 Welcome to the House Price Prediction Agent!")
        print("I can help you predict house price categories using machine learning.")
        print("Type 'quit' to exit, 'help' for assistance, or 'model info' for model details.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    help_text = """
                    I can help you predict house prices! Here's how:
                    
                    🏠 Example questions you can ask:
                    - "Predict the price for a 3-bedroom, 2-bathroom house with 1500 sq ft, 10 years old"
                    - "What price category would a new 4-bedroom house with 2500 sq ft be?"
                    - "Tell me about your prediction model"
                    - "What factors affect house prices?"
                    
                    📊 I need these details for predictions:
                    - Number of bedrooms (1-5)
                    - Number of bathrooms (1-3) 
                    - Square footage (800-3500)
                    - Age in years (0-50)
                    """
                    print(help_text)
                    continue
                
                if not user_input:
                    continue
                
                print("\n🤖 Agent: ", end="")
                response = self.chat(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}\n")

def main():
    """Main function to run the agent"""
    print("🚀 Starting House Price Prediction Agent...")
    
    try:
        # Initialize agent
        agent = HousePriceAgent()
        
        # Run interactive session
        agent.run_interactive_session()
        
    except Exception as e:
        print(f"❌ Failed to start agent: {e}")

if __name__ == "__main__":
    main()