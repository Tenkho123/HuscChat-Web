// start: Sidebar
document.querySelector('.chat-sidebar-profile-toggle').addEventListener('click', function(e) {
    e.preventDefault()
    this.parentElement.classList.toggle('active')
})

document.addEventListener('click', function(e) {
    if(!e.target.matches('.chat-sidebar-profile, .chat-sidebar-profile *')) {
        document.querySelector('.chat-sidebar-profile').classList.remove('active')
    }
})
// end: Sidebar

// start: Chat list
document.addEventListener("DOMContentLoaded", () => {
    const createChatButton = document.getElementById("create-new-chat");
    const messageList = document.querySelector(".content-messages-list");
    const conversationContainer = document.querySelector(".chat-content"); // or use a wrapper if available

    let chatCounter = 3; // Start at 3 if you already have 2 conversations

    createChatButton.addEventListener("click", (e) => {
        e.preventDefault();

        const conversationId = `conversation-${chatCounter}`;

        // ✅ Create new <li> chat item
        const newListItem = document.createElement("li");
        newListItem.innerHTML = `
            <a href="#" data-conversation="#${conversationId}">
                <img class="content-message-image" src="https://upload.wikimedia.org/wikipedia/commons/9/91/%C4%90%E1%BA%A1i_h%E1%BB%8Dc_khoa_h%E1%BB%8Dc_logo_-_HUSC_01.jpg" alt="">
                <span class="content-message-info">
                    <span class="content-message-name">HuscGPT</span>
                    <span class="content-message-text"></span>
                </span>
                <span class="content-message-more">
                    <!-- <span class="content-message-unread">1</span> -->
                    <span class="content-message-time">${new Date().getHours()}:${new Date().getMinutes().toString().padStart(2, "0")}</span>
                </span>
            </a>
        `;
        messageList.insertBefore(newListItem, messageList.children[1]); // insert at top after "Chats" title

        // ✅ Create new conversation DOM
        const newConversation = document.createElement("div");
        newConversation.classList.add("conversation");
        newConversation.id = conversationId;
        newConversation.style.display = "none"; // hidden by default

        newConversation.innerHTML = `
        <div class="conversation-top">
                <button type="button" class="conversation-back"><i class="ri-arrow-left-line"></i></button>
                <div class="conversation-user">
                    <img class="conversation-user-image" src="https://upload.wikimedia.org/wikipedia/commons/9/91/%C4%90%E1%BA%A1i_h%E1%BB%8Dc_khoa_h%E1%BB%8Dc_logo_-_HUSC_01.jpg" alt="">
                    <div>
                        <div class="conversation-user-name">HuscGPT</div>
                        <div class="conversation-user-status online">online</div>
                    </div>
                </div>
                <div class="conversation-buttons">
                    <button type="button"><i class="ri-phone-fill"></i></button>
                    <button type="button"><i class="ri-vidicon-line"></i></button>
                    <button type="button"><i class="ri-information-line"></i></button>
                </div>
            </div>
            <div class="conversation-main">
                <ul class="conversation-wrapper">
                    <div class="coversation-divider"><span>Today</span></div>
                    <li class="conversation-item me">
                    <!--
                        <div class="conversation-item-content">
                            <div class="conversation-item-box">
                                <div class="conversation-item-text">
                                    <p>This is a new conversation.</p>
                                    <div class="conversation-item-time">12:30</div>
                                </div>
                            </div>
                        </div>
                    -->
                    </li>
                </ul>
            </div>
            <div class="conversation-form">
                <button type="button" class="conversation-form-button"><i class="ri-emotion-line"></i></button>
                <div class="conversation-form-group">
                    <textarea class="conversation-form-input" rows="1" placeholder="Type here..."></textarea>
                    <button type="button" class="conversation-form-record"><i class="ri-mic-line"></i></button>
                </div>
                <button type="button" class="conversation-form-button conversation-form-submit"><i class="ri-send-plane-2-line"></i></button>
            </div>
        `;

        // Append the new conversation to the container
        conversationContainer.appendChild(newConversation);

        chatCounter++;
    });

    // 🧠 Event delegation for clicking on a chat item
    messageList.addEventListener("click", (e) => {
        const anchor = e.target.closest("a[data-conversation]");
        if (!anchor) return;

        e.preventDefault();
        const targetSelector = anchor.getAttribute("data-conversation");
        const targetConversation = document.querySelector(targetSelector);

        if (!targetConversation) return;

        // Hide all existing conversations
        document.querySelectorAll(".conversation").forEach(conv => {
            conv.style.display = "none";
        });

        // Show selected conversation
        targetConversation.style.display = "block";
    });
});


document.querySelector(".chat-content").addEventListener("click", function (e) {
    const sendBtn = e.target.closest(".conversation-form-submit");
    if (!sendBtn) return;

    const form = sendBtn.closest(".conversation");
    const textarea = form.querySelector(".conversation-form-input");

    sendMessage(textarea);
});

document.querySelector(".chat-content").addEventListener("keydown", function (e) {
    const textarea = e.target.closest(".conversation-form-input");
    if (!textarea) return;

    if (e.key === "Enter") {
        if (e.shiftKey || e.ctrlKey) {
            return; // Allow newline
        }

        e.preventDefault(); // Prevent default newline
        sendMessage(textarea);
    }
});

function sendMessage(textarea) {
    const form = textarea.closest(".conversation");
    const inputField = form.querySelector(".conversation-form-input");
    const messageText = inputField.value.trim();
    const scrollContainer = form.querySelector(".conversation-main");
    if (!messageText) return;

    const messageList = form.querySelector(".conversation-wrapper");

    const newMessage = document.createElement("li");
    newMessage.classList.add("conversation-item");
    newMessage.innerHTML = `
        <div class="conversation-item-content">
            <div class="conversation-item-box">
                <div class="conversation-item-text">
                    <p>${messageText}</p>
                    <div class="conversation-item-time">${getCurrentTime()}</div>
                </div>
            </div>
        </div>
    `;

    messageList.appendChild(newMessage);
    inputField.value = "";
    scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: "smooth"
    });

    const isFirstMessage = messageList.querySelectorAll(".conversation-item").length === 1;

    if (!isFirstMessage) {
        
        const conversationId = form.id; // e.g. conversation-3
        
        const conversationSelector = `#${conversationId}`;

        const previewLink = document.querySelector(`a[data-conversation="${conversationSelector}"]`);
        
        if (previewLink) {
            const previewText = previewLink.querySelector(".content-message-text");
            console.log(previewText);
            if (previewText) {
                previewText.textContent = messageText;
            }
        }
    }

    const botMessage = document.createElement("li");
        botMessage.classList.add("conversation-item", "bot");
        botMessage.innerHTML = `
            <div class="conversation-item-content">
                <div class="conversation-item-box">
                    <div class="conversation-item-text">
                        <p>"hi"</p>
                        <div class="conversation-item-time">${getCurrentTime()}</div>
                    </div>
                </div>
            </div>
        `;
        messageList.appendChild(botMessage);
        scrollContainer.scrollTo({
            top: scrollContainer.scrollHeight,
            behavior: "smooth"
        });
}

function getCurrentTime() {
    const now = new Date();
    return now.getHours().toString().padStart(2, "0") + ":" + now.getMinutes().toString().padStart(2, "0");
}

function openConversation(conversationId) {
    // Hide all conversations
    document.querySelectorAll(".conversation").forEach(conv => {
        conv.style.display = "none";
    });

    // Hide the default message if it's visible
    const defaultConv = document.querySelector(".conversation-default");
    if (defaultConv) defaultConv.style.display = "none";

    // Show the selected conversation
    const target = document.querySelector(`#${conversationId}`);
    if (target) {
        target.style.display = "block";
    }
}

// Get current chat number
function getCurrentChatCounter() {
    const activeConversation = document.querySelector(".conversation:not([style*='display: none'])");
    if (!activeConversation) return null;
    
    const match = activeConversation.id.match(/conversation-(\d+)/);
    return match ? parseInt(match[1]) : null;
}
// end: Chat list

// start: Coversation
document.querySelectorAll('.conversation-item-dropdown-toggle').forEach(function(item) {
    item.addEventListener('click', function(e) {
        e.preventDefault()
        if(this.parentElement.classList.contains('active')) {
            this.parentElement.classList.remove('active')
        } else {
            document.querySelectorAll('.conversation-item-dropdown').forEach(function(i) {
                i.classList.remove('active')
            })
            this.parentElement.classList.add('active')
        }
    })
})

document.addEventListener('click', function(e) {
    if(!e.target.matches('.conversation-item-dropdown, .conversation-item-dropdown *')) {
        document.querySelectorAll('.conversation-item-dropdown').forEach(function(i) {
            i.classList.remove('active')
        })
    }
})

document.querySelectorAll('.conversation-form-input').forEach(function(item) {
    item.addEventListener('input', function() {
        this.rows = this.value.split('\n').length
    })
})

document.querySelectorAll('[data-conversation]').forEach(function(item) {
    item.addEventListener('click', function(e) {
        e.preventDefault()
        document.querySelectorAll('.conversation').forEach(function(i) {
            i.classList.remove('active')
        })
        document.querySelector(this.dataset.conversation).classList.add('active')
    })
})

document.querySelectorAll('.conversation-back').forEach(function(item) {
    item.addEventListener('click', function(e) {
        e.preventDefault()
        this.closest('.conversation').classList.remove('active')
        document.querySelector('.conversation-default').classList.add('active')
    })
})
// end: Coversation
