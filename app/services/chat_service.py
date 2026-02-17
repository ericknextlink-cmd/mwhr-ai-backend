import json
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from difflib import get_close_matches
from app.core.config import settings

class ChatService:
    def __init__(self):
        self.openai_api_key = settings.OPENAI_API_KEY
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.chat_data_path = os.path.join(self.data_dir, "chat-data.json")
        self.knowledge_base_path = os.path.join(self.data_dir, "guidelines.md")
        self.pattern_guide = self._load_pattern_guide()
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_pattern_guide(self) -> Dict[str, Any]:
        try:
            with open(self.chat_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat data: {e}")
            return {"intents": [], "default_response": "I apologize, but I am having trouble accessing my knowledge base."}

    def _load_knowledge_base(self) -> str:
        try:
            with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return ""

    def _try_pattern_match(self, user_message: str) -> Optional[str]:
        user_message_lower = user_message.lower().strip()
        
        # Skip for long messages
        if len(user_message_lower) > 100:
            return None
            
        is_likely_greeting = any(word in user_message_lower for word in ["hi", "hello", "hey", "good morning", "good afternoon", "greetings"])
        
        all_patterns = []
        pattern_map = {}
        
        for intent in self.pattern_guide.get("intents", []):
            if intent["id"] == "greeting" and not is_likely_greeting:
                continue
                
            for pattern in intent["patterns"]:
                pattern_lower = pattern.lower()
                all_patterns.append(pattern_lower)
                pattern_map[pattern_lower] = intent["response"]
        
        # Fuzzy matching using difflib
        matches = get_close_matches(user_message_lower, all_patterns, n=1, cutoff=0.65)
        
        if matches:
            return pattern_map[matches[0]]
            
        return None

    _OFF_TOPIC_RESPONSE = (
        "I can only assist with questions about the Ministry of Works, Housing & Water Resources "
        "certification and application process. For code or other topics, please try another platform."
    )

    def _is_off_topic_or_code(self, message: str) -> bool:
        """Heuristic: treat as off-topic if message looks like code or clearly unrelated to certification."""
        if not message or len(message.strip()) < 2:
            return True
        msg = message.strip()
        msg_lower = msg.lower()
        # Code blocks
        if "```" in msg or "def " in msg_lower or "function " in msg_lower or "import " in msg_lower:
            return True
        if "const " in msg_lower or "let " in msg_lower or "var " in msg_lower or "class " in msg_lower:
            return True
        if "=>" in msg or "->" in msg or "{" in msg and "}" in msg and ("(" in msg or ";" in msg):
            return True
        # Mostly non-letters (symbols/numbers)
        letters = sum(1 for c in msg if c.isalpha() or c.isspace())
        if len(msg) > 20 and letters / max(len(msg), 1) < 0.4:
            return True
        return False

    async def generate_response(self, message: str, history: List[Dict[str, str]]) -> str:
        # Step 1: Pattern Matching
        pattern_response = self._try_pattern_match(message)
        if pattern_response:
            return pattern_response

        # Step 2: Reject off-topic / code
        if self._is_off_topic_or_code(message):
            return self._OFF_TOPIC_RESPONSE
            
        # Step 3: AI Generation
        if not self.openai_api_key:
            return self.pattern_guide.get("default_response", "") + " (AI service unavailable)"
            
        chat_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=self.openai_api_key
        )
        
        fee_responses = "\n\n".join([
            i["response"] for i in self.pattern_guide.get("intents", []) 
            if i["id"].startswith("fees_")
        ])
        
        system_prompt = f"""You are Mavis, a helpful assistant for the Ministry of Works, Housing & Water Resources (MWHWR) in Ghana. 
Your role is to provide accurate information about contractor classification and certification processes.

CRITICAL INSTRUCTIONS:
1. You have access to TWO information sources:
   - KNOWLEDGE BASE: Contains detailed guidelines, procedures, and general information
   - PATTERN GUIDE: Contains specific data like fees, contact info, and quick reference responses
   
2. ALWAYS use BOTH sources to answer questions:
   - For general information, procedures, requirements: Use the KNOWLEDGE BASE
   - For specific fees, contact details, quick facts: Use the PATTERN GUIDE
   - Combine information from both sources when answering complex questions

3. When answering fee questions:
   - The PATTERN GUIDE contains the exact fee amounts - USE THESE
   - Reference the KNOWLEDGE BASE for context about fee structure and validity periods
   - Never say fees are not available - they are in the PATTERN GUIDE below

4. Answer format:
   - Be comprehensive and cite information from both sources
   - Use the exact fee amounts from the PATTERN GUIDE
   - Reference the KNOWLEDGE BASE for procedures and requirements
   - Be professional, accurate, and helpful

5. If information is truly not in either source, acknowledge this and provide what you can from available sources

6. If the user's message is primarily code, or clearly unrelated to certification/ministry (e.g. general coding help, math, other topics), respond with exactly: "I can only assist with questions about the Ministry of Works, Housing & Water Resources certification and application process. For code or other topics, please try another platform."

KNOWLEDGE BASE (Guidelines and Procedures):
{self.knowledge_base}

PATTERN GUIDE (Specific Data - Fees, Contact Info, Quick References):
{json.dumps(self.pattern_guide.get('intents', []), indent=2)}

SPECIFIC FEE INFORMATION (for quick reference):
{fee_responses}

Remember: You are representing an official government ministry. Use ALL available information sources to provide complete, accurate answers."""

        messages = [SystemMessage(content=system_prompt)]
        
        # Add history (limit to last 5 interactions)
        for msg in history[-5:]:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            else:
                messages.append(AIMessage(content=msg.get("content", "")))
                
        messages.append(HumanMessage(content=message))
        
        response = await chat_model.ainvoke(messages)
        return str(response.content)

chat_service = ChatService()
