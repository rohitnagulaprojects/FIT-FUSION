// Chatbot functionality
function toggleChatbot() {
    const chatbot = document.getElementById("chatbot-container");
    chatbot.classList.toggle("hidden");
    
    // If opening the chatbot and no messages exist yet, show welcome message
    if (!chatbot.classList.contains("hidden")) {
        const messagesContainer = document.getElementById("chatbot-messages");
        if (messagesContainer.children.length === 0) {
            // Add welcome message
            addMessage("Bot: Welcome to My Fitness Club! How can I help you with your fitness journey today?", "bot");
        }
    }
}

// Ensure the close button works properly
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("close-chatbot").addEventListener("click", function () {
        toggleChatbot();
    });

    // Add event listener for the Enter key in the input field
    document.getElementById("user-message").addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
});

// Add this function to display messages in the chat
function addMessage(message, sender) {
    const chatMessages = document.getElementById("chatbot-messages");
    const messageElement = document.createElement("div");
    messageElement.classList.add("message");
    messageElement.classList.add(sender + "-message");
    
    // Split the message to separate the sender prefix (e.g., "You: " or "Bot: ")
    const messageParts = message.split(": ", 2);
    
    if (messageParts.length > 1) {
        const senderLabel = document.createElement("span");
        senderLabel.classList.add("sender-label");
        senderLabel.textContent = messageParts[0] + ": ";
        
        messageElement.appendChild(senderLabel);
        messageElement.appendChild(document.createTextNode(messageParts[1]));
    } else {
        messageElement.textContent = message;
    }
    
    chatMessages.appendChild(messageElement);
    
    // Auto-scroll to the bottom of the chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
    const userInput = document.getElementById("user-message");
    const messageText = userInput.value.trim();

    if (messageText === "") return;

    addMessage("You: " + messageText, "user");
    userInput.value = "";

    // Display typing indicator
    const typingIndicator = document.createElement("div");
    typingIndicator.id = "typing-indicator";
    typingIndicator.classList.add("message", "bot-message", "typing");
    typingIndicator.innerHTML = "Bot is typing<span class='dot-1'>.</span><span class='dot-2'>.</span><span class='dot-3'>.</span>";
    document.getElementById("chatbot-messages").appendChild(typingIndicator);

    fetch("/chat", {
        method: "POST",
        body: JSON.stringify({ message: messageText }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        return response.json();
    })
    .then(data => {
        // Remove typing indicator
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();
        
        if (data.response) {
            addMessage("Bot: " + data.response, "bot");
        } else {
            addMessage("Bot: Unexpected response format.", "bot");
        }
    })
    .catch(error => {
        // Remove typing indicator
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();
        
        console.error("Error:", error);
        addMessage("Bot: Error fetching response. Try again later.", "bot");
    });
}