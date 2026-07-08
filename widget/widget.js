(function() {
    // 1. Inject CSS
    const style = document.createElement('style');
    style.innerHTML = `
        #chat-widget-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            font-family: Arial, sans-serif;
        }
        #chat-trigger-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        #chat-window {
            display: none;
            flex-direction: column;
            width: 350px;
            height: 500px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        #chat-header {
            background: #007bff;
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: bold;
        }
        #close-chat-btn {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
        }
        #chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background: #f9f9f9;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .message {
            padding: 10px 15px;
            border-radius: 15px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .bot-msg {
            background: #e9ecef;
            align-self: flex-start;
            border-bottom-left-radius: 2px;
        }
        .user-msg {
            background: #007bff;
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 2px;
        }
        #chat-input-area {
            display: flex;
            padding: 10px;
            border-top: 1px solid #ddd;
            background: white;
        }
        #chat-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 20px;
            outline: none;
        }
        #send-btn {
            background: none;
            border: none;
            color: #007bff;
            font-size: 20px;
            cursor: pointer;
            padding: 0 10px;
        }
    `;
    document.head.appendChild(style);

    // Inject HTML Structure
    const container = document.createElement('div');
    container.id = 'chat-widget-container';
    container.innerHTML = `
        <button id="chat-trigger-btn">💬</button>
        <div id="chat-window">
            <div id="chat-header">
                <span>Ügyfélszolgálati Asszisztens</span>
                <button id="close-chat-btn">×</button>
            </div>
            <div id="chat-messages">
                <div class="message bot-msg">Üdvözlöm! Miben segíthetek ma?</div>
            </div>
            <div id="chat-input-area">
                <input type="text" id="chat-input" placeholder="Írja be a kérdését..." autofocus autocomplete="off">
                <button id="send-btn">➤</button>
            </div>
        </div>
    `;
    document.body.appendChild(container);

    // JavaScript Logic
    const chatTriggerBtn = document.getElementById('chat-trigger-btn');
    const chatWindow = document.getElementById('chat-window');
    const closeChatBtn = document.getElementById('close-chat-btn');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');

    const API_URL = globalThis.API_BASE_URL + "/chat"; // Change to your actual backend URL later
    let chatHistory = [];

    chatTriggerBtn.addEventListener('click', () => {
        chatWindow.style.display = 'flex';
        chatTriggerBtn.style.display = 'none';
    });

    closeChatBtn.addEventListener('click', () => {
        chatWindow.style.display = 'none';
        chatTriggerBtn.style.display = 'flex';
    });

    function appendMessage(text, senderClass) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', senderClass);
        msgDiv.textContent = text;
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user-msg');
        chatInput.value = '';

        const typingMsg = document.createElement('div');
        typingMsg.classList.add('message', 'bot-msg');
        typingMsg.textContent = "Gondolkodik...";
        typingMsg.id = "typing-indicator";
        chatMessages.appendChild(typingMsg);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, history: chatHistory })
            });

            const data = await response.json();
            document.getElementById('typing-indicator').remove();

            if (response.ok) {
                appendMessage(data.reply, 'bot-msg');
                chatHistory.push({ role: "user", content: text }, { role: "assistant", content: data.reply });

                if (chatHistory.length > 10) {
                    chatHistory = chatHistory.slice(-10);
                }
            } else {
                appendMessage("Hiba történt a szerver oldalon.", 'bot-msg');
            }
        } catch (error) {
            console.error("Error connecting to API:", error);
            document.getElementById('typing-indicator')?.remove();
            appendMessage("Hálózati hiba. Nem tudok csatlakozni a szerverhez.", 'bot-msg');
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
})();
