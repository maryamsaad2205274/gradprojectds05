import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Users,
  Activity,
  Upload,
  CalendarClock,
  ArrowUpRight,
  TrendingUp,
  MessageCircle,
  Check,
} from "lucide-react";
import { getMessages, markRead, seedIfEmpty, subscribe, type PatientMessage } from "@/lib/patientMessages";


export const Route = createFileRoute("/_app/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard — DentAlign" }] }),
  component: Dashboard,
});

const stats = [
  { label: "Active Patients", value: "248", delta: "+12 this month", icon: Users, tone: "primary" },
  { label: "Cases Analyzed", value: "1,842", delta: "+184 this week", icon: Activity, tone: "primary" },
  { label: "Pending Uploads", value: "7", delta: "Needs review", icon: Upload, tone: "warning" },
  { label: "Appointments Today", value: "14", delta: "3 remaining", icon: CalendarClock, tone: "primary" },
];

const recentCases = [
  { name: "Jana", id: "PT-1042", type: "Class II", status: "Analyzed", date: "Today, 10:24" },
  { name: "Yassin", id: "PT-1041", type: "Open bite", status: "In review", date: "Today, 09:10" },
  { name: "Maryam", id: "PT-1040", type: "Crowding", status: "Analyzed", date: "Yesterday" },
  { name: "Bassant", id: "PT-1039", type: "Class III", status: "Pending", date: "Yesterday" },
  { name: "Sama", id: "PT-1038", type: "Overjet", status: "Analyzed", date: "2 days ago" },
];

const upcoming = [
  { time: "11:00", name: "Jana", reason: "Bracket check" },
  { time: "12:30", name: "Maryam", reason: "Initial consult" },
  { time: "14:15", name: "Bassant", reason: "Aligner fitting" },
  { time: "16:00", name: "Yassin", reason: "Progress photos" },
];

function timeAgo(ts: number) {
  const s = Math.floor((Date.now() - ts) / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function Dashboard() {
  const [messages, setMessages] = useState<PatientMessage[]>([]);
  useEffect(() => {
    seedIfEmpty();
    setMessages(getMessages());
    return subscribe(() => setMessages(getMessages()));
  }, []);
  const unread = messages.filter((m) => !m.read).length;

  return (
    <div className="space-y-8">

      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-primary">Good morning, Dr. Sama</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight">Clinical overview</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Here's what's happening across your practice today.
          </p>
        </div>
        <Button
          asChild
          className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
        >
          <Link to="/upload">
            <Upload className="mr-2 h-4 w-4" /> New case upload
          </Link>
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((s) => (
          <Card key={s.label} className="border-border/60 shadow-[var(--shadow-card)]">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <s.icon className="h-5 w-5" />
                </div>
                <Badge variant="secondary" className="bg-primary/5 text-primary border-0">
                  <TrendingUp className="mr-1 h-3 w-3" /> live
                </Badge>
              </div>
              <p className="mt-5 text-3xl font-bold tracking-tight">{s.value}</p>
              <p className="text-sm font-medium text-foreground/80">{s.label}</p>
              <p className="mt-1 text-xs text-muted-foreground">{s.delta}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="border-border/60 shadow-[var(--shadow-card)] lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle className="text-base">Recent cases</CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">Latest analyses across your roster</p>
            </div>
            <Button asChild variant="ghost" size="sm">
              <Link to="/patients">
                View all <ArrowUpRight className="ml-1 h-3 w-3" />
              </Link>
            </Button>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-border/60">
              {recentCases.map((c) => (
                <div
                  key={c.id}
                  className="flex items-center justify-between gap-4 py-3 text-sm"
                >
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                      {c.name.split(" ").map((n) => n[0]).join("")}
                    </div>
                    <div>
                      <p className="font-medium">{c.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {c.id} · {c.type}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <Badge
                      variant="secondary"
                      className={
                        c.status === "Analyzed"
                          ? "bg-primary/10 text-primary border-0"
                          : c.status === "Pending"
                          ? "bg-amber-500/10 text-amber-700 border-0"
                          : "bg-muted text-muted-foreground border-0"
                      }
                    >
                      {c.status}
                    </Badge>
                    <span className="hidden text-xs text-muted-foreground md:inline">{c.date}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <CardTitle className="text-base">Today's schedule</CardTitle>
            <p className="text-xs text-muted-foreground">Upcoming appointments</p>
          </CardHeader>
          <CardContent className="space-y-3">
            {upcoming.map((u) => (
              <div
                key={u.time}
                className="flex items-center gap-3 rounded-lg border border-border/60 p-3"
              >
                <div className="flex h-12 w-12 flex-col items-center justify-center rounded-md bg-primary/10 text-primary">
                  <span className="text-[10px] uppercase">Pm</span>
                  <span className="text-sm font-semibold leading-none">{u.time}</span>
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{u.name}</p>
                  <p className="truncate text-xs text-muted-foreground">{u.reason}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/60 shadow-[var(--shadow-card)]">
        <CardHeader className="flex flex-row items-center justify-between space-y-0">
          <div>
            <div className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4 text-primary" />
              <CardTitle className="text-base">Patient messages</CardTitle>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Short questions submitted by patients through their portal.
            </p>
          </div>
          {unread > 0 && (
            <Badge variant="secondary" className="bg-primary/10 text-primary border-0">
              {unread} new
            </Badge>
          )}
        </CardHeader>
        <CardContent>
          {messages.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">No patient questions yet.</p>
          ) : (
            <div className="divide-y divide-border/60">
              {messages.map((m) => (
                <div key={m.id} className="flex items-start gap-3 py-4">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                    {m.patient.split(" ").map((n) => n[0]).join("")}
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-sm font-medium">{m.patient}</p>
                      <span className="text-xs text-muted-foreground">{m.patientId}</span>
                      <span className="text-xs text-muted-foreground">· {timeAgo(m.createdAt)}</span>
                      {!m.read && (
                        <Badge variant="secondary" className="bg-primary/10 text-primary border-0 text-[10px]">
                          New
                        </Badge>
                      )}
                    </div>
                    <p className="mt-1 text-sm text-foreground/80">"{m.question}"</p>
                  </div>
                  {!m.read && (
                    <Button size="sm" variant="ghost" onClick={() => markRead(m.id)}>
                      <Check className="mr-1 h-3 w-3" /> Mark read
                    </Button>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>

  );
}
