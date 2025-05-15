import { Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ChatButton({ onClick }: { onClick: () => void }) {
  return (
    <Button
      onClick={onClick}
      className="fixed bottom-6 right-6 h-12 w-12 rounded-full shadow-lg bg-primary hover:bg-primary/90 transition-transform transform hover:scale-105"
    >
      <Sparkles className="h-6 w-6" />
    </Button>
  );
}
