import { createFileRoute, Link } from "@tanstack/react-router";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Download, Share2, CheckCircle2, Sparkles, Lock, StickyNote, Save, CalendarClock, KeyRound, Copy, RefreshCw, Eye, EyeOff, Send, GitCompare, Images } from "lucide-react";
import { toast } from "sonner";
import { useRef, useState } from "react";

export const Route = createFileRoute("/_app/analysis")({
  head: () => ({ meta: [{ title: "Analysis — DentAlign" }] }),
  component: AnalysisPage,
});

const measurements = [
  { label: "SNA angle", value: "82.4°", norm: "82° ± 2", status: "normal" },
  { label: "SNB angle", value: "78.1°", norm: "80° ± 2", status: "low" },
  { label: "ANB angle", value: "4.3°", norm: "2° ± 2", status: "high" },
  { label: "FMA", value: "26.8°", norm: "25° ± 4", status: "normal" },
  { label: "U1 to SN", value: "108.2°", norm: "103° ± 5", status: "high" },
  { label: "Overjet", value: "5.2 mm", norm: "2 mm", status: "high" },
  { label: "Overbite", value: "3.8 mm", norm: "2 mm", status: "normal" },
  { label: "Facial convexity", value: "12°", norm: "8° ± 4", status: "high" },
];

const tone: Record<string, string> = {
  normal: "bg-emerald-500/10 text-emerald-700",
  high: "bg-amber-500/10 text-amber-700",
  low: "bg-sky-500/10 text-sky-700",
};

// SVG landmark overlay placeholder
function LandmarkOverlay({ view }: { view: "frontal" | "lateral" }) {
  const points =
    view === "frontal"
      ? [
          { x: 50, y: 22, l: "Gl" },
          { x: 50, y: 38, l: "N" },
          { x: 38, y: 40, l: "Or-R" },
          { x: 62, y: 40, l: "Or-L" },
          { x: 50, y: 55, l: "Pr" },
          { x: 50, y: 72, l: "Sto" },
          { x: 50, y: 88, l: "Me" },
          { x: 30, y: 55, l: "Zy-R" },
          { x: 70, y: 55, l: "Zy-L" },
        ]
      : [
          { x: 38, y: 20, l: "G" },
          { x: 42, y: 32, l: "N" },
          { x: 60, y: 48, l: "Pr" },
          { x: 58, y: 60, l: "Sn" },
          { x: 56, y: 72, l: "Ls" },
          { x: 54, y: 82, l: "Li" },
          { x: 46, y: 92, l: "Pog" },
          { x: 40, y: 88, l: "B" },
          { x: 30, y: 70, l: "Go" },
        ];

  return (
    <div className="relative aspect-[4/5] overflow-hidden rounded-xl border border-border/60 bg-gradient-to-br from-muted/60 to-muted/30">
      {/* face silhouette */}
      <svg viewBox="0 0 100 125" className="absolute inset-0 h-full w-full">
        <defs>
          <linearGradient id="face" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="oklch(0.95 0.02 230)" />
            <stop offset="100%" stopColor="oklch(0.88 0.04 235)" />
          </linearGradient>
        </defs>
        {view === "frontal" ? (
          <ellipse cx="50" cy="60" rx="28" ry="38" fill="url(#face)" stroke="oklch(0.75 0.05 235)" strokeWidth="0.4" />
        ) : (
          <path
            d="M 32 20 Q 28 50 36 80 Q 42 100 48 105 L 60 100 Q 64 80 62 60 Q 60 35 50 22 Z"
            fill="url(#face)"
            stroke="oklch(0.75 0.05 235)"
            strokeWidth="0.4"
          />
        )}
        {/* connecting lines */}
        {view === "lateral" && (
          <polyline
            points={points.map((p) => `${p.x},${p.y}`).join(" ")}
            fill="none"
            stroke="oklch(0.55 0.18 245)"
            strokeWidth="0.3"
            strokeDasharray="1 1"
          />
        )}
        {/* landmark points */}
        {points.map((p) => (
          <g key={p.l}>
            <circle cx={p.x} cy={p.y} r="1.6" fill="oklch(0.55 0.18 245)" stroke="white" strokeWidth="0.4" />
            <text
              x={p.x + 2.5}
              y={p.y + 1.2}
              fontSize="2.2"
              fill="oklch(0.3 0.1 245)"
              fontWeight="600"
            >
              {p.l}
            </text>
          </g>
        ))}
      </svg>
      <div className="absolute left-3 top-3 rounded-md bg-background/90 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-foreground backdrop-blur">
        {view} · {points.length} landmarks
      </div>
    </div>
  );
}

type CaseStatus = "New" | "Pending" | "In Review" | "Analyzed" | "Active" | "Retention";

const statusTone: Record<CaseStatus, string> = {
  New: "bg-emerald-500/10 text-emerald-700",
  Pending: "bg-amber-500/10 text-amber-700",
  "In Review": "bg-sky-500/10 text-sky-700",
  Analyzed: "bg-primary/10 text-primary",
  Active: "bg-violet-500/10 text-violet-700",
  Retention: "bg-muted text-muted-foreground",
};

const allStatuses: CaseStatus[] = ["New", "Pending", "In Review", "Analyzed", "Active", "Retention"];

// ---- Photo timeline / comparison ---------------------------------------------------
type TimelinePhoto = { id: string; label: string; date: string; hue: number };

const timelinePhotos: TimelinePhoto[] = [
  { id: "t0", label: "Initial", date: "Feb 12, 2026", hue: 18 },
  { id: "t6", label: "Month 3", date: "May 11, 2026", hue: 28 },
  { id: "t12", label: "Month 6", date: "Aug 14, 2026", hue: 38 },
  { id: "t18", label: "Month 12", date: "Feb 09, 2027", hue: 48 },
];

// Placeholder "photo" — a stylised face SVG that shifts subtly per timepoint so the
// comparison slider visibly changes. Real uploads would replace this.
function FacePlaceholder({ photo, view = "front" }: { photo: TimelinePhoto; view?: "front" | "side" }) {
  const skin = `oklch(0.86 0.06 ${photo.hue})`;
  const shade = `oklch(0.72 0.08 ${photo.hue + 10})`;
  // jaw shifts upward / forward over time to suggest treatment progress
  const t = timelinePhotos.findIndex((p) => p.id === photo.id);
  const jawY = 88 - t * 1.2;
  const lipW = 14 - t * 1.5;
  return (
    <svg viewBox="0 0 100 125" className="h-full w-full">
      <defs>
        <linearGradient id={`bg-${photo.id}`} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="oklch(0.96 0.02 230)" />
          <stop offset="100%" stopColor="oklch(0.9 0.03 235)" />
        </linearGradient>
      </defs>
      <rect width="100" height="125" fill={`url(#bg-${photo.id})`} />
      {view === "front" ? (
        <>
          <ellipse cx="50" cy="62" rx="26" ry="36" fill={skin} stroke={shade} strokeWidth="0.5" />
          <ellipse cx="40" cy="55" rx="2.4" ry="1.4" fill="oklch(0.3 0.08 245)" />
          <ellipse cx="60" cy="55" rx="2.4" ry="1.4" fill="oklch(0.3 0.08 245)" />
          <path d={`M ${50 - lipW / 2} 78 Q 50 ${82 - t} ${50 + lipW / 2} 78`} fill="none" stroke="oklch(0.55 0.12 25)" strokeWidth="1.2" strokeLinecap="round" />
          <ellipse cx="50" cy={jawY} rx="14" ry="6" fill={shade} opacity="0.4" />
        </>
      ) : (
        <>
          <path
            d={`M 34 22 Q 28 50 36 80 Q 42 ${jawY + 12} 50 ${jawY + 5} L 62 ${jawY} Q 64 78 62 58 Q 60 32 50 22 Z`}
            fill={skin}
            stroke={shade}
            strokeWidth="0.5"
          />
          <ellipse cx="52" cy="52" rx="1.8" ry="1.2" fill="oklch(0.3 0.08 245)" />
          <path d={`M 56 ${74 - t} Q 60 ${76 - t} 60 ${78 - t}`} fill="none" stroke="oklch(0.55 0.12 25)" strokeWidth="1.2" strokeLinecap="round" />
        </>
      )}
      <text x="6" y="118" fontSize="5" fontWeight="700" fill="oklch(0.35 0.05 245)">{photo.label}</text>
      <text x="6" y="124" fontSize="3.2" fill="oklch(0.5 0.03 245)">{photo.date}</text>
    </svg>
  );
}

function ComparisonSlider({ left, right, view }: { left: TimelinePhoto; right: TimelinePhoto; view: "front" | "side" }) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const setFromClientX = (clientX: number) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const p = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.max(0, Math.min(100, p)));
  };

  return (
    <div
      ref={ref}
      onPointerDown={(e) => {
        dragging.current = true;
        (e.target as Element).setPointerCapture?.(e.pointerId);
        setFromClientX(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && setFromClientX(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      className="relative aspect-[4/5] w-full cursor-ew-resize select-none overflow-hidden rounded-xl border border-border/60 bg-muted/30"
    >
      <div className="absolute inset-0">
        <FacePlaceholder photo={right} view={view} />
      </div>
      <div className="absolute inset-y-0 left-0 overflow-hidden" style={{ width: `${pos}%` }}>
        <div className="absolute inset-y-0 left-0" style={{ width: ref.current?.clientWidth ?? "100%" }}>
          <FacePlaceholder photo={left} view={view} />
        </div>
      </div>
      <div className="pointer-events-none absolute inset-y-0" style={{ left: `${pos}%` }}>
        <div className="absolute inset-y-0 -ml-px w-0.5 bg-primary shadow-[0_0_0_2px_oklch(1_0_0_/_0.4)]" />
        <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg">
          <GitCompare className="h-4 w-4" />
        </div>
      </div>
      <div className="absolute left-3 top-3 rounded-md bg-background/90 px-2 py-1 text-[10px] font-bold uppercase tracking-wider backdrop-blur">
        {left.label}
      </div>
      <div className="absolute right-3 top-3 rounded-md bg-background/90 px-2 py-1 text-[10px] font-bold uppercase tracking-wider backdrop-blur">
        {right.label}
      </div>
    </div>
  );
}

function PhotoTimeline() {
  const [leftId, setLeftId] = useState(timelinePhotos[0].id);
  const [rightId, setRightId] = useState(timelinePhotos[timelinePhotos.length - 1].id);
  const [view, setView] = useState<"front" | "side">("front");
  const left = timelinePhotos.find((p) => p.id === leftId)!;
  const right = timelinePhotos.find((p) => p.id === rightId)!;

  return (
    <Card className="border-border/60 shadow-[var(--shadow-card)]">
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <Images className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Photo timeline & comparison</CardTitle>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Drag the divider to compare two visits side-by-side. Pick any two timepoints below.
            </p>
          </div>
          <Tabs value={view} onValueChange={(v) => setView(v as "front" | "side")}>
            <TabsList className="h-8">
              <TabsTrigger value="front" className="text-xs">Front</TabsTrigger>
              <TabsTrigger value="side" className="text-xs">Side</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-[1fr_280px]">
          <ComparisonSlider left={left} right={right} view={view} />
          <div className="space-y-3">
            <div>
              <p className="text-xs font-medium text-muted-foreground">Left side of slider</p>
              <Select value={leftId} onValueChange={setLeftId}>
                <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {timelinePhotos.map((p) => (
                    <SelectItem key={p.id} value={p.id}>{p.label} — {p.date}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <p className="text-xs font-medium text-muted-foreground">Right side of slider</p>
              <Select value={rightId} onValueChange={setRightId}>
                <SelectTrigger className="mt-1"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {timelinePhotos.map((p) => (
                    <SelectItem key={p.id} value={p.id}>{p.label} — {p.date}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs text-foreground/80">
              Comparing <span className="font-semibold text-primary">{left.label}</span> with{" "}
              <span className="font-semibold text-primary">{right.label}</span> · {view} view
            </div>
          </div>
        </div>

        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Full timeline</p>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {timelinePhotos.map((p) => {
              const active = p.id === leftId || p.id === rightId;
              return (
                <button
                  key={p.id}
                  onClick={() => (p.id === leftId ? null : setRightId(p.id))}
                  className={`group overflow-hidden rounded-lg border bg-card text-left transition ${
                    active ? "border-primary ring-2 ring-primary/30" : "border-border/60 hover:border-primary/50"
                  }`}
                >
                  <div className="aspect-square">
                    <FacePlaceholder photo={p} view={view} />
                  </div>
                  <div className="p-2">
                    <p className="text-xs font-semibold">{p.label}</p>
                    <p className="text-[10px] text-muted-foreground">{p.date}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}


function AnalysisPage() {
  const [notes, setNotes] = useState(
    "Discuss elastics compliance with patient.\nCheck molar relationship at next visit.",
  );
  const [status, setStatus] = useState<CaseStatus>("Analyzed");

  // Next appointment
  const [apptDate, setApptDate] = useState("");
  const [apptTime, setApptTime] = useState("");
  const [apptReason, setApptReason] = useState("Bracket check");
  const [scheduled, setScheduled] = useState<{ date: string; time: string; reason: string } | null>(null);

  // Patient portal access code
  const generateCode = () => Math.random().toString(36).slice(2, 8).toUpperCase();
  const [accessCode, setAccessCode] = useState("EM4K9P");
  const [showCode, setShowCode] = useState(false);

  const saveAppointment = () => {
    if (!apptDate || !apptTime) {
      toast.error("Pick a date and time first");
      return;
    }
    setScheduled({ date: apptDate, time: apptTime, reason: apptReason });
    toast.success("Appointment scheduled · patient portal updated");
  };


  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className={`${statusTone[status]} border-0`}>
              <CheckCircle2 className="mr-1 h-3 w-3" /> {status}
            </Badge>
            <span className="text-xs text-muted-foreground">Generated 2 min ago</span>
          </div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            Jana <span className="text-muted-foreground font-medium text-xl">· PT-1042</span>
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Class II Div. 1 · 14y · Initial diagnostic record
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select value={status} onValueChange={(v) => { setStatus(v as CaseStatus); toast.success(`Status updated to ${v}`); }}>
            <SelectTrigger className="h-9 w-[150px] text-xs font-medium">
              <SelectValue placeholder="Set status" />
            </SelectTrigger>
            <SelectContent>
              {allStatuses.map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline">
            <Share2 className="mr-2 h-4 w-4" /> Share with patient
          </Button>
          <Button
            onClick={() => toast.success("Report PDF generated")}
            className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
          >
            <Download className="mr-2 h-4 w-4" /> Download report
          </Button>
        </div>
      </div>

      <Tabs defaultValue="visual">
        <TabsList>
          <TabsTrigger value="visual">Visual analysis</TabsTrigger>
          <TabsTrigger value="measurements">Measurements</TabsTrigger>
          <TabsTrigger value="ai">AI insights</TabsTrigger>
        </TabsList>

        <TabsContent value="visual" className="mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card className="border-border/60 shadow-[var(--shadow-card)]">
              <CardHeader>
                <CardTitle className="text-base">Frontal view — landmarks</CardTitle>
              </CardHeader>
              <CardContent>
                <LandmarkOverlay view="frontal" />
                <p className="mt-3 text-xs text-muted-foreground">
                  9 facial landmarks detected with 98.2% confidence.
                </p>
              </CardContent>
            </Card>
            <Card className="border-border/60 shadow-[var(--shadow-card)]">
              <CardHeader>
                <CardTitle className="text-base">Lateral view — cephalometric</CardTitle>
              </CardHeader>
              <CardContent>
                <LandmarkOverlay view="lateral" />
                <p className="mt-3 text-xs text-muted-foreground">
                  Soft tissue profile traced and analyzed.
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="measurements" className="mt-6">
          <Card className="border-border/60 shadow-[var(--shadow-card)]">
            <CardContent className="p-6">
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                {measurements.map((m) => (
                  <div key={m.label} className="rounded-lg border border-border/60 bg-card p-4">
                    <p className="text-xs uppercase tracking-wider text-muted-foreground">
                      {m.label}
                    </p>
                    <p className="mt-2 text-2xl font-bold tracking-tight">{m.value}</p>
                    <div className="mt-2 flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">Norm {m.norm}</span>
                      <Badge variant="secondary" className={`${tone[m.status]} border-0 capitalize`}>
                        {m.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="ai" className="mt-6">
          <Card className="border-border/60 shadow-[var(--shadow-card)]">
            <CardContent className="space-y-5 p-6">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <h3 className="font-semibold">Treatment recommendations</h3>
              </div>
              {[
                { label: "Skeletal pattern", v: 78, t: "Mild Class II skeletal base with retrognathic mandible." },
                { label: "Dental relationship", v: 65, t: "Increased overjet (5.2mm) with proclined maxillary incisors." },
                { label: "Soft tissue profile", v: 70, t: "Convex profile; lip competence reduced at rest." },
              ].map((i) => (
                <div key={i.label}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{i.label}</span>
                    <span className="text-xs text-muted-foreground">Confidence {i.v}%</span>
                  </div>
                  <Progress value={i.v} className="mt-2" />
                  <p className="mt-2 text-sm text-muted-foreground">{i.t}</p>
                </div>
              ))}
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-4 text-sm">
                <p className="font-medium text-primary">Suggested approach</p>
                <p className="mt-1 text-foreground/80">
                  Consider functional appliance therapy followed by fixed orthodontics.
                  Re-evaluate growth at 6-month intervals.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <PhotoTimeline />

      <div className="grid gap-6 lg:grid-cols-2">

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <div className="flex items-center gap-2">
              <CalendarClock className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Next appointment</CardTitle>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Schedule the patient's next visit. Saving this updates their patient portal automatically.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground">Date</label>
                <Input type="date" value={apptDate} onChange={(e) => setApptDate(e.target.value)} className="mt-1" />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground">Time</label>
                <Input type="time" value={apptTime} onChange={(e) => setApptTime(e.target.value)} className="mt-1" />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Reason</label>
              <Select value={apptReason} onValueChange={setApptReason}>
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["Bracket check", "Wire change", "Elastics review", "Bonding", "Debonding", "Retention check", "Records / photos", "Consultation"].map((r) => (
                    <SelectItem key={r} value={r}>{r}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {scheduled && (
              <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm">
                <p className="font-medium text-primary">Scheduled · synced to portal</p>
                <p className="mt-1 text-foreground/80">
                  {scheduled.date} at {scheduled.time} — {scheduled.reason}
                </p>
              </div>
            )}
            <div className="flex justify-end">
              <Button
                size="sm"
                onClick={saveAppointment}
                className="bg-[image:var(--gradient-primary)] hover:opacity-95"
              >
                <Send className="mr-2 h-4 w-4" /> Save & notify patient
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <div className="flex items-center gap-2">
              <KeyRound className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Patient portal access code</CardTitle>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Share this code with the patient so they can log into their portal to upload photos and view their plan.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center gap-2">
              <div className="flex-1 rounded-lg border border-border/60 bg-muted/40 px-4 py-3 font-mono text-2xl font-bold tracking-[0.3em] text-primary">
                {showCode ? accessCode : "••••••"}
              </div>
              <Button size="icon" variant="outline" onClick={() => setShowCode((s) => !s)}>
                {showCode ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  navigator.clipboard.writeText(accessCode);
                  toast.success("Code copied to clipboard");
                }}
              >
                <Copy className="mr-2 h-4 w-4" /> Copy
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setAccessCode(generateCode());
                  setShowCode(true);
                  toast.success("New access code generated");
                }}
              >
                <RefreshCw className="mr-2 h-4 w-4" /> Regenerate
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Regenerating immediately invalidates the previous code.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/60 shadow-[var(--shadow-card)]">

        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <StickyNote className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">My private notes</CardTitle>
            </div>
            <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
              <Lock className="h-3 w-3" /> Only you can see this
            </span>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            Personal reminders about this case. Never shared with the patient or included in reports.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Jot down anything you want to remember about this case…"
            className="min-h-[140px] resize-y"
          />
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">Auto-saved locally · last edit just now</p>
            <Button
              size="sm"
              variant="outline"
              onClick={() => toast.success("Private notes saved")}
            >
              <Save className="mr-2 h-4 w-4" /> Save notes
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
