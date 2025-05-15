import React, { useEffect, useState, useRef } from "react";
import { Send, X, Paperclip } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import chatbotIcon from "./chatbot.png";

interface Message {
  id: string;
  sender: string;
  text?: string;
  imageUrl?: string;
  showButtons?: boolean;
}

interface ChatPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatPanel({ isOpen, onClose }: ChatPanelProps) {
  // Chat States
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [posterDescription, setPosterDescription] = useState("");
  const [conversationHistory, setConversationHistory] = useState<string[]>([]);
  const [pendingEscalation, setPendingEscalation] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [helpOpen, setHelpOpen] = useState(false);

  // Auto-scroll Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  };

  // Initialize chat with welcome message
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([
        {
          id: Date.now().toString(),
          sender: "bot",
          text: "Hello there, I am Hera! How may I help you today?",
        },
      ]);
    }
  }, [isOpen, messages.length]);

  const handleImageChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      e.target.value = "";
      setSelectedImage(file);
    }
  };

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: input || "",
      imageUrl: selectedImage ? URL.createObjectURL(selectedImage) : undefined,
    };
    setMessages((prev) => [...prev, userMessage]);
    setConversationHistory((prev) => [...prev, input]);
    setInput("");

    if (pendingEscalation) {
      handleEscalationResponse(input);
      return;
    }

    try {
      let extractedDescription = posterDescription;

      if (selectedImage) {
        const formData = new FormData();
        formData.append("image", selectedImage);

        const response = await fetch("http://localhost:5001/api/extract-text", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        extractedDescription = data.poster_description || "";
        setPosterDescription(extractedDescription);
      }

      const hrResponse = await fetch("http://localhost:5001/api/hr-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: input,
          poster_description: extractedDescription,
          conversation_history: conversationHistory,
        }),
      });

      const result = await hrResponse.json();

      if (hrResponse.status === 202) {
        setPendingEscalation(input);
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            sender: "bot",
            text: "I couldn't find a direct answer. Do you want to escalate this to HR? (Yes/No)",
            showButtons: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            sender: "bot",
            text: result.answer || "No response from the bot.",
          },
        ]);
      }
    } catch (error) {
      console.error("Error processing message:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: "bot",
          text: "Failed to process the message.",
        },
      ]);
    }

    setSelectedImage(null);
  };

  const handleEscalationResponse = (response: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        sender: "user",
        text: response === "yes" ? "Yes" : "No",
      },
    ]);

    if (response === "yes") {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: "bot",
          text: "Your question has been escalated to HR. How else may I help you?",
        },
      ]);
      escalateToHR(pendingEscalation);
    } else if (response === "no") {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: "bot",
          text: "Alright, I won't escalate this. Let me know if you need anything else.",
        },
      ]);
    }

    setPendingEscalation(null);
  };

  const escalateToHR = async (question: string | null) => {
    if (!question) return;

    try {
      const response = await fetch("http://localhost:5001/api/escalate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        console.error("Failed to escalate question to HR.");
      }
    } catch (error) {
      console.error("Error escalating question to HR:", error);
    }
  };

  return (
    <div
      className={cn(
        "fixed inset-0 h-full w-full bg-white shadow-lg flex flex-col transition-opacity duration-300",
        isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
      )}
    >
      {/* Top Bar */}
      <div
        className="flex items-center justify-center relative p-3 w-full"
        style={{
          background: "linear-gradient(to bottom, #0A75C2, rgba(255, 255, 255, 0))",
          height: "75px",
          backdropFilter: "blur(8px)",
          WebkitBackdropFilter: "blur(8px)",
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
        }}
      >
        <button
          onClick={onClose}
          className="absolute left-4 p-2 bg-white rounded-full shadow-md hover:shadow-lg transition"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-gray-600"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            viewBox="0 0 24 24"
          >
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
        </button>

        <div className="flex items-center space-x-3">
          <img
            src={chatbotIcon}
            alt="Chatbot"
            className="h-10 w-10 rounded-full"
          />
          <h2 className="text-3xl font-extrabold text-white drop-shadow-lg">
            Chat Hera
          </h2>
        </div>

        <button
          onClick={() => setHelpOpen(true)}
          className="absolute right-4 p-2 bg-white rounded-full shadow-md hover:shadow-lg transition"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-6 w-6 text-gray-600"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="1" />
            <circle cx="12" cy="5" r="1" />
            <circle cx="12" cy="19" r="1" />
          </svg>
        </button>
      </div>

      {/* Help Panel */}
      <div
        className={`fixed top-0 right-0 h-full w-2/5 bg-white shadow-lg transform ${helpOpen ? "translate-x-0" : "translate-x-full"
          } transition-transform duration-300 ease-in-out z-50`}
      >
        <div className="p-4 flex items-center justify-between border-b">
          <h2 className="text-lg font-semibold">Help & Support</h2>
          <button
            onClick={() => setHelpOpen(false)}
            className="p-2 bg-white rounded-full shadow-md hover:shadow-lg transition"
          >
            <X className="h-6 w-6 text-gray-600" />
          </button>
        </div>

        <div className="p-4 text-gray-700 h-[75vh]">
          <p>Need assistance? Learn More About ChatHera:</p>
          <div className="mt-5 bg-white shadow-lg rounded-lg border border-gray-200 h-[65vh] overflow-y-auto p-6">
            <ul className="space-y-6">
              <li>
                <h3 className="text-xl font-bold text-blue-700">1. About ChatHera</h3>
                <p className="text-gray-700 mt-2">
                  Answers all your HR-related questions. Example:
                </p>
                <ul className="ml-5 list-disc text-gray-600 mt-2">
                  <li><strong>Medical & Insurance Policies:</strong> Understand policies, claim processes, and eligibility.</li>
                  <li><strong>Work Systems:</strong> Learn how to clock in/out, request leaves, or access internal tools.</li>
                </ul>
              </li>
              <li>
                <h3 className="text-xl font-bold text-green-700">2. How It Works</h3>
                <p className="text-gray-700 mt-2">
                  Just type your question, and the ChatHera takes care of the rest:
                </p>
                <ul className="ml-5 list-disc text-gray-600 mt-2">
                  <li>But what if no answer was found? No problem! You will be provided with an option to escalate to HR.</li>
                  <li>Choose "Yes" to escalate or "No" to not escalate</li>
                  <li>The moment HR responds, you'll be notified via Email.</li>
                  <li>Next time, find the same answer through ChatHera (Given that it is not confidential).</li>
                </ul>
              </li>
              <li>
                <h3 className="text-xl font-bold text-purple-700"> 3. Features – Beyond Just a Chatbot!</h3>
                <p className="text-gray-700 mt-2">
                  This isn't just a chatbot—it's a powerful HR assistant packed with smart features.
                </p>
                <ul className="ml-5 list-disc text-gray-600 mt-2">
                  <li> Instant Answers – Get real-time responses for all your HR-related concerns.</li>
                  <li> HR Escalation System – Directly forward unresolved queries to HR with just one click.</li>
                  <li> Image-Based Queries – Upload pictures of HR posters, policies, or forms and get related answers instantly.</li>
                </ul>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Today Divider */}
      <div className="text-center text-gray-500 text-sm my-6 pb-4 bg-white shadow-md rounded-md">
        Today
      </div>

      {/* Messages Area */}
      <ScrollArea
        ref={scrollAreaRef}
        className="flex-1 p-4"
      >
        <div className="flex flex-col">
          {messages.map((message) => {
            const isUser = message.sender.toLowerCase() !== "bot";
            return (
              <div
                key={message.id}
                className={`flex w-full mb-4 ${isUser ? "justify-end" : "justify-start"
                  }`}
              >
                {!isUser && (
                  <img
                    src={chatbotIcon}
                    alt="Chatbot"
                    className="h-8 w-8 rounded-full mr-2"
                  />
                )}
                <div className="flex flex-col max-w-[75%]">
                  <div
                    className={`p-3 rounded-3xl shadow-md ${isUser
                      ? "bg-[#0078C1] text-white px-5 py-3 text-base font-medium rounded-2xl max-w-lg ml-auto text-left"
                      : "bg-[#F3F4F6] text-gray-900 px-5 py-3 text-base font-medium rounded-2xl max-w-lg text-left"
                      }`}
                  >
                    {message.text && (
                      <div className="prose">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.text}
                        </ReactMarkdown>
                      </div>
                    )}
                    {message.imageUrl && (
                      <img
                        src={message.imageUrl}
                        alt="Uploaded"
                        className="max-w-full rounded-lg mt-2"
                      />
                    )}
                  </div>

                  {message.showButtons && (
                    <div className="flex gap-2 mt-2">
                      <button
                        onClick={() => handleEscalationResponse("yes")}
                        className="rounded-full px-4 py-2 text-sm font-medium border border-blue-500 bg-white/50 backdrop-blur-sm hover:bg-blue-100/50 transition-all shadow-[0_0_8px_rgba(0,120,193,0.3)]"
                      >
                        Yes
                      </button>
                      <button
                        onClick={() => handleEscalationResponse("no")}
                        className="rounded-full px-4 py-2 text-sm font-medium border border-blue-500 bg-white/50 backdrop-blur-sm hover:bg-blue-100/50 transition-all shadow-[0_0_8px_rgba(0,120,193,0.3)]"
                      >
                        No
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>

      {/* Quick Prompts */}
      <div className="flex flex-wrap justify-center gap-2 px-4 py-2">
        {["How to clock in?", "What is the IT Ticket system?", "How to clock out?"].map(
          (text, idx) => (
            <Button
              key={idx}
              variant="outline"
              className="rounded-full px-4 py-2 text-sm font-medium border border-blue-500 bg-white/50 backdrop-blur-sm hover:bg-blue-100/50 transition-all shadow-[0_0_8px_rgba(0,120,193,0.3)]"
              onClick={() => {
                setInput(text);
                setTimeout(scrollToBottom, 100);
              }}
            >
              {text}
            </Button>
          )
        )}
      </div>

      {/* Input Area */}
      <div className="border-t p-4 bg-white">
        {selectedImage && (
          <div className="mb-2 flex items-center space-x-2">
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Preview"
              className="h-12 w-12 rounded-lg object-cover"
            />
            <Button variant="ghost" size="icon" onClick={() => setSelectedImage(null)}>
              🗑️
            </Button>
          </div>
        )}

        <div className="flex items-center gap-3 border border-gray-300 rounded-full px-4 py-3 shadow-md">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => document.getElementById("chat-file-input")?.click()}
            className="bg-white border border-gray-300 text-gray-700 rounded-full h-12 w-12 flex items-center justify-center shadow-sm hover:bg-gray-100 transition"
          >
            <Paperclip className="h-5 w-5" />
          </Button>
          <input
            id="chat-file-input"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="hidden"
          />

          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter") {
                handleSend();
                setTimeout(scrollToBottom, 100);
              }
            }}
            placeholder="Type a message..."
            className="flex-1 bg-transparent text-black placeholder-gray-500 border-none focus:ring-0 text-lg"
          />

          <Button
            onClick={() => {
              handleSend();
              setTimeout(scrollToBottom, 100);
            }}
            className="bg-white border border-gray-300 text-gray-700 rounded-full h-12 w-12 flex items-center justify-center shadow-sm hover:bg-gray-100 transition"
          >
            <Send className="h-6 w-6" />
          </Button>
        </div>
      </div>
    </div>
  );
}