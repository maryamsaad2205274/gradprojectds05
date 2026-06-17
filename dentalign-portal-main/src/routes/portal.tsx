import { createFileRoute, Link } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import {
  Sparkles,
  ShieldCheck,
  FileText,
  Download,
  Calendar,
  ArrowLeft,
  ImagePlus,
  X,
  CheckCircle2,
  Sun,
  Camera,
  User,
  CloudUpload,
  MessageCircle,
  Send,
} from "lucide-react";
import { toast } from "sonner";
import { addMessage } from "@/lib/patientMessages";


export const Route = createFileRoute("/portal")({
  head: () => ({
    meta: [
      { title: "Patient Portal — DentAlign" },
      { name: "description", content: "Access your orthodontic treatment records securely." },
    ],
  }),
  component: PortalPage,
});

function PortalPage() {
  const [code, setCode] = useState("");
  const [authed, setAuthed] = useState(false);

  if (authed) return <PortalDashboard onExit={() => setAuthed(false)} />;

  return (
    <div className="flex min-h-screen items-center justify-center bg-[image:var(--gradient-soft)] p-6">
      <div className="w-full max-w-md">
        <Link
          to="/login"
          className="mb-6 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" /> Back to clinician sign-in
        </Link>

        <Card className="border-border/60 shadow-[var(--shadow-elegant)]">
          <CardContent className="p-8">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[image:var(--gradient-primary)] text-primary-foreground">
                <Sparkles className="h-5 w-5" />
              </div>
              <div>
                <p className="text-base font-bold">DentAlign Portal</p>
                <p className="text-xs text-muted-foreground">For patients</p>
              </div>
            </div>

            <h1 className="mt-8 text-2xl font-semibold tracking-tight">
              Enter your secure access code
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              We sent a 6-digit code to the email on file with your clinic.
            </p>

            <div className="mt-8 flex justify-center">
              <InputOTP maxLength={6} value={code} onChange={setCode}>
                <InputOTPGroup>
                  {[0, 1, 2, 3, 4, 5].map((i) => (
                    <InputOTPSlot key={i} index={i} className="h-12 w-12 text-lg" />
                  ))}
                </InputOTPGroup>
              </InputOTP>
            </div>

            <Button
              disabled={code.length !== 6}
              onClick={() => setAuthed(true)}
              className="mt-8 h-11 w-full bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
            >
              <ShieldCheck className="mr-2 h-4 w-4" /> Access my records
            </Button>

            <p className="mt-6 text-center text-xs text-muted-foreground">
              Don't have a code?{" "}
              <a href="#" className="font-medium text-primary hover:underline">
                Contact your clinic
              </a>
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function PhotoDrop({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: string | null;
  onChange: (v: string | null) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);

  const handleFile = (file?: File) => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    onChange(url);
  };

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <Label className="text-sm font-medium">{label}</Label>
        {value && (
          <button
            onClick={() => onChange(null)}
            className="text-xs text-muted-foreground hover:text-destructive flex items-center gap-1"
          >
            <X className="h-3 w-3" /> Remove
          </button>
        )}
      </div>
      <div
        onClick={() => ref.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFile(e.dataTransfer.files?.[0]);
        }}
        className="group relative flex aspect-[4/5] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-border/80 bg-muted/30 transition hover:border-primary/60 hover:bg-primary/5"
      >
        {value ? (
          <img src={value} alt={label} className="absolute inset-0 h-full w-full object-cover" />
        ) : (
          <>
            <div className="flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary transition group-hover:scale-105">
              <ImagePlus className="h-6 w-6" />
            </div>
            <p className="mt-3 text-sm font-medium">Tap or drop photo here</p>
            <p className="mt-1 px-4 text-center text-xs text-muted-foreground">{hint}</p>
          </>
        )}
        <input
          ref={ref}
          type="file"
          accept="image/*"
          capture="user"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? undefined)}
        />
      </div>
    </div>
  );
}

const photoRules = [
  { icon: Sun, text: "Take the photo in a well-lit room — natural daylight works best." },
  { icon: Camera, text: "Keep the camera centered and at eye level — no tilting up or down." },
  { icon: User, text: "Face neutral, lips relaxed, hair pulled away from your face." },
  { icon: CheckCircle2, text: "Plain background (a wall is perfect). Remove glasses and face masks." },
];

function PortalDashboard({ onExit }: { onExit: () => void }) {
  const [front, setFront] = useState<string | null>(null);
  const [side, setSide] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [sentRecently, setSentRecently] = useState<string | null>(null);

  const submit = () => {
    if (!front || !side) {
      toast.error("Please add both the front and side photos.");
      return;
    }
    toast.success("Photos sent to your clinic.");
    setFront(null);
    setSide(null);
  };

  const sendQuestion = () => {
    const trimmed = question.trim();
    if (trimmed.length < 5) {
      toast.error("Write a short question first (at least a few words).");
      return;
    }
    if (trimmed.length > 280) {
      toast.error("Please keep it under 280 characters.");
      return;
    }
    addMessage({ patient: "Jana", patientId: "PT-1042", question: trimmed });
    setSentRecently(trimmed);
    setQuestion("");
    toast.success("Question sent to Dr. Sama.");
  };


  return (
    <div className="min-h-screen bg-[image:var(--gradient-soft)]">
      <header className="border-b border-border/60 bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[image:var(--gradient-primary)] text-primary-foreground">
              <Sparkles className="h-5 w-5" />
            </div>
            <span className="font-bold">DentAlign Portal</span>
          </div>
          <Button variant="ghost" size="sm" onClick={onExit}>
            Sign out
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 p-6">
        <div>
          <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
            Secure session · expires in 30 min
          </Badge>
          <h1 className="mt-3 text-3xl font-bold tracking-tight">Hello, Jana</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Your orthodontic treatment with Dr. Sama
          </p>
        </div>

        <Card className="border-primary/30 bg-[image:var(--gradient-soft)] shadow-[var(--shadow-card)]">
          <CardContent className="flex flex-wrap items-center justify-between gap-4 p-5">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Calendar className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider text-muted-foreground">
                  Your next appointment
                </p>
                <p className="mt-0.5 text-lg font-semibold">Jun 8, 2026 · 14:30</p>
                <p className="text-sm text-muted-foreground">Reason: Bracket check</p>
              </div>
            </div>
            <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
              Set by Dr. Sama
            </Badge>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-3">
          {[
            { label: "Treatment progress", value: "Month 4 of 18", icon: Calendar },
            { label: "Last visit", value: "May 11, 2026", icon: Calendar },
            { label: "Reports available", value: "3 documents", icon: FileText },
          ].map((s) => (
            <Card key={s.label} className="border-border/60 shadow-[var(--shadow-card)]">
              <CardContent className="p-5">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <s.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-wider text-muted-foreground">
                      {s.label}
                    </p>
                    <p className="mt-0.5 font-semibold">{s.value}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>


        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle className="text-base">Send your progress photos</CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">
                  Your clinic uses these for your next check-up. Please read the tips below first.
                </p>
              </div>
              <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
                2 photos needed
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="rounded-xl border border-primary/20 bg-primary/5 p-4">
              <p className="text-sm font-semibold text-primary">How to take a good photo</p>
              <ul className="mt-3 grid gap-3 sm:grid-cols-2">
                {photoRules.map((r) => (
                  <li key={r.text} className="flex items-start gap-2 text-sm text-foreground/80">
                    <r.icon className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
                    <span>{r.text}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
              <PhotoDrop
                label="Front view"
                hint="Look straight at the camera. Keep your face centered in the frame, with both ears visible."
                value={front}
                onChange={setFront}
              />
              <PhotoDrop
                label="Side view (profile)"
                hint="Turn your head 90° to the side. Chin level, looking straight ahead — not up or down."
                value={side}
                onChange={setSide}
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border/60 bg-card p-4">
              <p className="text-xs text-muted-foreground">
                Photos are encrypted and shared only with your clinical team.
              </p>
              <Button
                onClick={submit}
                className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
              >
                <CloudUpload className="mr-2 h-4 w-4" /> Send to my clinic
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <div className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Ask Dr. Sama a question</CardTitle>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Short questions only — your doctor will see this on their dashboard. For emergencies please call the clinic.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. My elastics keep snapping after meals — should I switch sizes?"
              maxLength={280}
              className="min-h-[100px] resize-y"
            />
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">{question.length}/280 characters</p>
              <Button
                onClick={sendQuestion}
                className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
              >
                <Send className="mr-2 h-4 w-4" /> Send to my doctor
              </Button>
            </div>
            {sentRecently && (
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm">
                <p className="text-xs font-semibold text-primary">Sent · awaiting reply</p>
                <p className="mt-1 text-foreground/80">"{sentRecently}"</p>
              </div>
            )}
          </CardContent>
        </Card>


        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardContent className="p-6">
            <h2 className="text-base font-semibold">Your reports</h2>
            <div className="mt-4 divide-y divide-border/60">
              {[
                { name: "Initial diagnostic report", date: "Feb 12, 2026" },
                { name: "Mid-treatment progress", date: "Apr 18, 2026" },
                { name: "Latest cephalometric analysis", date: "May 20, 2026" },
              ].map((r) => (
                <div key={r.name} className="flex items-center justify-between py-3">
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
                      <FileText className="h-4 w-4" />
                    </div>
                    <div>
                      <p className="text-sm font-medium">{r.name}</p>
                      <p className="text-xs text-muted-foreground">{r.date}</p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="mr-2 h-4 w-4" /> PDF
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
