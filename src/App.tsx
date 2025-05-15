import { useState } from "react";
import { ChatPanel } from "./components/chat/ChatPanel";

function App() {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div
      className="min-h-screen w-full"
      style={{
        background: "linear-gradient(to bottom, #0078C1, #FFFFFF)",
        color: "#003366",
      }}
    >
      {/* Header */}
      <div className="header flex items-center justify-between bg-white bg-opacity-90 p-4">
        <img src="static/hoya.png" alt="HOYA" className="h-10" />
        <div className="menu flex gap-6 text-sm">
          <span>NEWS</span>
          <span>Business Domains +</span>
          <span>About HOYA +</span>
          <span>Sustainability / ESG +</span>
          <span>Investor Relations +</span>
        </div>
        <div className="icons text-base">
          <span>🔍</span>
        </div>
      </div>

      {/* News Section */}
      <div className="news-section mx-10 my-8">
        <div className="news-title text-xl font-bold mb-4">NEWS</div>
        <div className="news-box bg-white p-6 rounded-lg shadow-md flex justify-between">
          <div className="news-item flex flex-col items-center text-center w-1/3 p-4">
            <img src="static/hoya.png" alt="Hoya" className="w-24 mb-4" />
            <p className="font-bold">2024.12.09</p>
            <p>
              HOYA Corporation acquired the remaining shares of PLASMABIOTICS SAS (73.53KB) 📄
            </p>
          </div>
          <div className="news-item flex flex-col items-center text-center w-1/3 p-4">
            <img src="static/pic2.png" alt="Report" className="w-24 mb-4" />
            <p className="font-bold">2024.09.10</p>
            <p>Issued HOYA Integrated Report 2024 (73.53KB)</p>
          </div>
          <div className="news-item flex flex-col items-center text-center w-1/3 p-4">
            <img src="static/pic1.png" alt="TCFD" className="w-24 mb-4" />
            <p className="font-bold">2024.05.27</p>
            <p>TCFD Disclosure updated in 2024 (1.45MB) 📄</p>
          </div>
        </div>
      </div>

      {/* Corporate Mission Section */}
      <div className="corporate-mission text-center py-8 bg-white">
        <h1 className="text-5xl font-bold text-blue-800">WHAT WE DO</h1>
        <p className="text-2xl text-blue-900 mt-4">Innovating For a Better Tomorrow</p>
      </div>

      {/* New Section: HOYA Electronics Focus */}
      <div className="hoya-focus-section bg-white py-8 px-8">
        <div className="max-w-4xl mx-auto text-center">
<h3 className="text-3xl font-bold text-blue-800">HOYA’s Products</h3>
        <p className="text-lg text-gray-700 leading-relaxed">
          <ul>
              <li>Photomask: Glass plate onto which a circuit pattern is drawn</li>
              <li>High-end Optical & EUV Mask Blanks: Substrate for a photomask</li>
          </ul>
      </p>
        </div>
      </div>
      
      {/* Chatbot Button with Hover Text */}
      <div className="fixed bottom-8 right-8 group">
        <button
          onClick={() => setIsChatOpen(true)}
          className="bg-blue-800 text-white w-16 h-16 rounded-full flex items-center justify-center text-2xl shadow-lg hover:bg-blue-700 transition"
        >
          💬
        </button>
        {/* Hover Text */}
        <div className="absolute bottom-20 right-0 bg-white text-blue-600 text-sm px-4 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 border-border-blue-600 whitespace-nowrap">
          Chat with Hera
        </div>
      </div>

      {/* Chat Panel (Overlay) */}
      <ChatPanel isOpen={isChatOpen} onClose={() => setIsChatOpen(false)} />
    </div>
  );
}

export default App;