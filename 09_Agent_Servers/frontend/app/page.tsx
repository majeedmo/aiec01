"use client";

import { useState } from "react";
import { Cat } from "lucide-react";

import { Chat } from "@/components/chat";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const ASSISTANTS = [
  {
    id: "simple_agent",
    label: "Simple Agent",
    description: "Tool-calling agent without helpfulness check",
  },
  {
    id: "agent_with_helpfulness",
    label: "With Helpfulness",
    description: "Judge and retry loop after each response",
  },
] as const;

export default function Page() {
  const [assistantId, setAssistantId] = useState<string>(ASSISTANTS[0].id);
  const selected =
    ASSISTANTS.find((assistant) => assistant.id === assistantId) ??
    ASSISTANTS[0];

  return (
    <main className="flex h-dvh flex-col">
      <header className="border-b bg-background">
        <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Cat className="size-4" />
            </div>
            <div className="min-w-0 leading-tight">
              <p className="text-sm font-medium">Cat Health Agent</p>
              <p className="truncate text-xs text-muted-foreground">
                {selected.description}
              </p>
            </div>
          </div>

          <div className="flex shrink-0 flex-col gap-1.5">
            <Label
              htmlFor="assistant-select"
              className="text-xs font-normal text-muted-foreground"
            >
              Assistant
            </Label>
            <Select
              value={assistantId}
              onValueChange={(value) => setAssistantId(value ?? ASSISTANTS[0].id)}
            >
              <SelectTrigger id="assistant-select" size="sm" className="w-[11rem]">
                <SelectValue placeholder="Choose assistant" />
              </SelectTrigger>
              <SelectContent align="end">
                {ASSISTANTS.map((assistant) => (
                  <SelectItem key={assistant.id} value={assistant.id}>
                    {assistant.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </header>

      <Chat key={assistantId} assistantId={assistantId} />
    </main>
  );
}
