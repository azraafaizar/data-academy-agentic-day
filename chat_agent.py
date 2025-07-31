
#Task 5
import gradio as gr
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from sample_products import PRODUCT_DATA
from difflib import get_close_matches

# Mock database classes
class Order(BaseModel):
    order_id: str
    customer_email: str
    products: List[Dict]
    total_amount: float
    status: str
    order_date: datetime
    shipping_address: str

class RefundRequest(BaseModel):
    order_id: str
    reason: str
    amount: float
    status: str = "pending"

# Mock databases
ORDERS_DB = {
    "ORD-001": Order(
        order_id="ORD-001",
        customer_email="john@example.com",
        products=[{"name": "TechMart Pro Laptop", "price": 899.99, "quantity": 1}],
        total_amount=899.99,
        status="delivered",
        order_date=datetime.now() - timedelta(days=5),
        shipping_address="123 Main St, Anytown, USA"
    ),
    "ORD-002": Order(
        order_id="ORD-002",
        customer_email="jane@example.com",
        products=[{"name": "SmartPhone X Pro", "price": 799.99, "quantity": 1}],
        total_amount=799.99,
        status="shipped",
        order_date=datetime.now() - timedelta(days=2),
        shipping_address="456 Oak Ave, Somewhere, USA"
    )
}

class ProductKnowledgeBase:
    def __init__(self, products: List[Dict]):
        self.products = products

    def search_products(self, query: str, category: str = None, min_price: Optional[float] = None, max_price: Optional[float] = None, required_features: Optional[List[str]] = None) -> List[Dict]:
        """Simple keyword-based product search"""
       
        if not query:
            return[]
       
        query_lower = query.lower()
        results = []


        for product in self.products:
            name = product.get('name', '').lower()
            description = product.get('description', '').lower()
            features = [f.lower() for f in product.get('features', [])]
            category_match = product.get('category', '').lower()
            price = float(product.get('price', 0))

            searchable_tokens = [name, description] + features

            match_found = any(get_close_matches(query_lower, [token], n=1, cutoff=0.7) for token in searchable_tokens)

            if match_found:
                if category and category.lower() != category_match:
                    continue
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue

                if required_features:
                    if not all(feature.lower() in features for feature in required_features):
                        continue

                results.append(product)

            return results[:5]  # Return top 5 results

    def get_product_by_id(self, product_id: str) -> Dict:
        """Get specific product by ID"""
        for product in self.products:
            if product['id'] == product_id:
                return product
        return None

kb = ProductKnowledgeBase(PRODUCT_DATA)

REFUNDS_DB = {}

#Tool validatiion
def validate_order_id(order_id: str) -> bool:
    return isinstance(order_id, str) and order_id.startswith("ORD-") and len(order_id) >= 7

def validate_email(email: str) -> bool:
    return isinstance(email, str) and "@" in email and "." in email.split("@")[-1]

# Tool functions
def check_order_status(order_id: str) -> Dict:
    """Check the status of a customer order"""
    if not validate_order_id(order_id):
        return {"success": False, "message": "Invalid order ID format."}
    
    if not order_id:
        return {"found": False, "message": "No order ID provided."}
    
    if order_id in ORDERS_DB:
        order = ORDERS_DB[order_id]
        return {
            "found": True,
            "order_id": order.order_id,
            "status": order.status,
            "order_date": order.order_date.strftime("%Y-%m-%d"),
            "total_amount": order.total_amount,
            "products": order.products
        }
    return {"found": False, "message": "Order not found"}

def process_refund(order_id: str, reason: str, customer_email: str) -> Dict:
    """Process a refund request for an order"""
    if not validate_order_id(order_id):
        return {"success": False, "message": "Invalid order ID format."}
    
    if not validate_email(customer_email):
        return {"success": False, "message": "Invalid email format."}
    
    if any(x.order_id == order_id for x in REFUNDS_DB.values()):
        return {"success": False, "message": "A refund has already been requested for this order."}

    if order_id not in ORDERS_DB:
        return {"success": False, "message": "Order not found"}

    order = ORDERS_DB[order_id]

    # Verify customer email matches
    if order.customer_email.lower() != customer_email.lower():
        return {"success": False, "message": "Email does not match order records"}

    # Check if order is eligible for refund
    if order.status not in ["delivered", "shipped"]:
        return {"success": False, "message": "Order not eligible for refund"}

    # Refund request
    refund_id = f"REF-{len(REFUNDS_DB) + 1:03d}"
    refund = RefundRequest(
        order_id=order_id,
        reason=reason,
        amount=order.total_amount
    )

    REFUNDS_DB[refund_id] = refund

    return {
        "success": True,
        "refund_id": refund_id,
        "amount": order.total_amount,
        "message": "Refund request submitted successfully. You will receive confirmation within 2-3 business days."
    }

def update_shipping_address(order_id: str, new_address: str, customer_email: str) -> Dict:
    """Update shipping address for an order"""
    if not validate_order_id(order_id):
        return {"success": False, "message": "Invalid order ID format."}
    
    if not validate_email(customer_email):
        return {"success": False, "message": "Invalid email format."}
    
    if order_id not in ORDERS_DB:
        return {"success": False, "message": "Order not found"}

    order = ORDERS_DB[order_id]

    # Verify customer email
    if order.customer_email.lower() != customer_email.lower():
        return {"success": False, "message": "Email does not match order records"}

    # Check if order can be modified
    if order.status in ["delivered", "cancelled"]:
        return {"success": False, "message": "Cannot modify address for delivered or cancelled orders"}

    # Update address
    ORDERS_DB[order_id].shipping_address = new_address

    return {
        "success": True,
        "message": f"Shipping address updated successfully for order {order_id}",
        "new_address": new_address
    }

def get_refund_status(refund_id: str) -> Dict:
    """Check the status of a refund request"""
    if refund_id in REFUNDS_DB:
        refund = REFUNDS_DB[refund_id]
        return {
            "found": True,
            "refund_id": refund_id,
            "order_id": refund.order_id,
            "status": refund.status,
            "amount": refund.amount,
            "reason": refund.reason
        }
    return {"found": False, "message": "Refund request not found"}

def cancel_order(order_id: str, customer_email: str) -> Dict:
    """Cancel an order if allowed"""
    if not validate_order_id(order_id):
        return {"success": False, "message": "Invalid order ID format."}
    
    if not validate_email(customer_email):
        return {"success": False, "message": "Invalid email format."}

    if order_id not in ORDERS_DB:
        return {"success": False, "message": "Order not found."}

    order = ORDERS_DB[order_id]

    if order.customer_email.lower() != customer_email.lower():
        return {"success": False, "message": "Email does not match order records."}

    if order.status in ["delivered", "cancelled"]:
        return {"success": False, "message": f"Order cannot be cancelled (current status: {order.status})."}

    order.status = "cancelled"
    return {
        "success": True,
        "message": f"Order {order_id} has been successfully cancelled."
    }

def get_tracking_number(order_id: str, customer_email: str) -> Dict:
    """Provide tracking number for shipped orders"""
    if order_id not in ORDERS_DB:
        return {"success": False, "message": "Order not found."}

    order = ORDERS_DB[order_id]
    
    if order.customer_email.lower() != customer_email.lower():
        return {"success": False, "message": "Email does not match order records."}
    
    if order.status != "shipped":
        return {"success": False, "message": "Tracking number is only available for shipped orders."}
    
    tracking_number = f"TM-{order_id[-3:]}-TRACK"
    return {
        "success": True,
        "tracking_number": tracking_number,
        "carrier": "TechMart Express",
        "message": f"Tracking number for your order is {tracking_number}."
    }

def check_product_availability(product_name: str) -> Dict:
    """Check if a product is in stock"""
    results = kb.search_products(product_name)

    if not results:
        return {"found": False, "message": "No matching product found."}

    product = results[0]  # Assume first match is best
    return {
        "found": True,
        "product": product['name'],
        "in_stock": product['in_stock'],
        "message": f"{product['name']} is {'in stock' if product['in_stock'] else 'currently out of stock'}."
    }

# Enhanced system prompt for tool calling
TOOL_SYSTEM_PROMPT = """
You are an advanced customer service agent for TechMart with access to order management tools.

You can help customers with:
- Checking order status
- Processing refund requests
- Updating shipping addresses
- Checking refund status
- Cancelling orders
- Get tracking number for orders
- Checking product availabilities

When customers request these actions, use the appropriate tools to help them. Always ask for necessary information like order ID and email address for verification.

For refund requests, be empathetic and gather the reason for the refund to improve our services.

Use the following rules for better conversations:
- Maintain memory of the latest order ID and email mentioned by the customer.
- If the customer says "cancel it", "cancel this order", or "I want to cancel", use the last mentioned order details.
- Do not repeatedly ask for information the customer has already given.
- If the email is slightly misspelled, attempt basic correction (e.g., '.cpm' → '.com') but ask for confirmation.
- Be concise and helpful. Always confirm actions taken using the tool responses.
- If a tool returns an error, explain why and what the user can do next.

Store policies:
- 30-day return policy for most items
- Free shipping on orders over $50
- Refunds processed within 3-5 business days
- Address changes only possible before shipping
"""

# Create agent with tools
provider = GoogleProvider(api_key='AIzaSyDwhZrg7OXq1RjXzsVz3grhr7Aa9XgI7-Q')
model = GoogleModel('gemini-2.5-flash', provider=provider)

agent = Agent(
    model=model,
    system_prompt=TOOL_SYSTEM_PROMPT,
    tools=[
        check_order_status,
        process_refund,
        update_shipping_address,
        get_refund_status,
        cancel_order,
        get_tracking_number,
        check_product_availability
    ]
)
# --- Chat state ---
chat_history = []

# --- Chat Function ---
async def chat(user_message, history):
    history = history or []
    history.append({"role": "user", "content": user_message})  # use lowercase 'user'

    full_prompt = "\n".join(f"{item['role']}: {item['content']}" for item in history)

    # Call Gemini LLM using async Agent
    result = await agent.run(full_prompt)

    # Extract plain text from AgentRunResult
    response = result.output if hasattr(result, 'output') else str(result)

    history.append({"role": "assistant", "content": response})
    return history, history


# --- Gradio UI ---
with gr.Blocks(title="Gemini Chatbot") as demo:
    gr.Markdown("## 🤖 Gemini Chatbot with Pydantic-AI + Gradio")
    
    chatbot = gr.Chatbot(type='messages')
    msg = gr.Textbox(placeholder="Type your message here...", label="Your Message")
    clear_btn = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state])
    clear_btn.click(lambda: ([], []), None, [chatbot, state])

# --- Launch the app ---
demo.queue().launch()
 