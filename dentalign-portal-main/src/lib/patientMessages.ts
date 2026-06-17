// Lightweight client-side store for patient → doctor messages.
// Backed by localStorage so the portal and dashboard share state across tabs.

export type PatientMessage = {
  id: string;
  patient: string;
  patientId: string;
  question: string;
  createdAt: number;
  read: boolean;
};

const KEY = "dentalign.patient_messages";
const EVT = "dentalign:messages-updated";

function read(): PatientMessage[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]") as PatientMessage[];
  } catch {
    return [];
  }
}

function write(list: PatientMessage[]) {
  localStorage.setItem(KEY, JSON.stringify(list));
  window.dispatchEvent(new Event(EVT));
}

export function getMessages(): PatientMessage[] {
  return read().sort((a, b) => b.createdAt - a.createdAt);
}

export function addMessage(input: { patient: string; patientId: string; question: string }) {
  const list = read();
  list.push({
    id: Math.random().toString(36).slice(2, 10),
    patient: input.patient,
    patientId: input.patientId,
    question: input.question.trim(),
    createdAt: Date.now(),
    read: false,
  });
  write(list);
}

export function markRead(id: string) {
  write(read().map((m) => (m.id === id ? { ...m, read: true } : m)));
}

export function subscribe(cb: () => void) {
  const handler = () => cb();
  window.addEventListener(EVT, handler);
  window.addEventListener("storage", handler);
  return () => {
    window.removeEventListener(EVT, handler);
    window.removeEventListener("storage", handler);
  };
}

// Seed a couple of sample messages on first load so the dashboard isn't empty.
export function seedIfEmpty() {
  if (typeof window === "undefined") return;
  if (read().length > 0) return;
  write([
    {
      id: "seed1",
      patient: "Yassin",
      patientId: "PT-1041",
      question: "My lower wire feels a bit loose on the right side, is that normal until my next visit?",
      createdAt: Date.now() - 1000 * 60 * 60 * 3,
      read: false,
    },
    {
      id: "seed2",
      patient: "Maryam",
      patientId: "PT-1040",
      question: "Can I switch to clear elastics for my sister's wedding next weekend?",
      createdAt: Date.now() - 1000 * 60 * 60 * 26,
      read: true,
    },
  ]);
}
