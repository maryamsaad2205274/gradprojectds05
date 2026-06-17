import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Search, Plus, Filter, MoreHorizontal } from "lucide-react";

export const Route = createFileRoute("/_app/patients")({
  head: () => ({ meta: [{ title: "Patients — DentAlign" }] }),
  component: PatientsPage,
});

type PatientStatus = "New" | "Pending" | "In Review" | "Analyzed" | "Active" | "Retention";

interface Patient {
  id: string;
  name: string;
  age: number;
  treatment: string;
  lastVisit: string;
  status: PatientStatus;
}

const initialPatients: Patient[] = [
  { id: "PT-1042", name: "Jana", age: 14, treatment: "Class II - Phase 1", lastVisit: "Today", status: "Active" },
  { id: "PT-1041", name: "Yassin", age: 27, treatment: "Open bite correction", lastVisit: "Today", status: "Analyzed" },
  { id: "PT-1040", name: "Maryam", age: 19, treatment: "Crowding (clear aligners)", lastVisit: "Yesterday", status: "Active" },
  { id: "PT-1039", name: "Bassant", age: 12, treatment: "Class III interceptive", lastVisit: "Yesterday", status: "In Review" },
  { id: "PT-1038", name: "Sama", age: 31, treatment: "Overjet reduction", lastVisit: "2 days", status: "Pending" },
  { id: "PT-1037", name: "Jana", age: 16, treatment: "Fixed appliance", lastVisit: "3 days", status: "Active" },
  { id: "PT-1036", name: "Maryam", age: 9, treatment: "Initial consult", lastVisit: "4 days", status: "New" },
  { id: "PT-1035", name: "Bassant", age: 22, treatment: "Aligner retention", lastVisit: "1 week", status: "Retention" },
];

const statusTone: Record<PatientStatus, string> = {
  New: "bg-emerald-500/10 text-emerald-700",
  Pending: "bg-amber-500/10 text-amber-700",
  "In Review": "bg-sky-500/10 text-sky-700",
  Analyzed: "bg-primary/10 text-primary",
  Active: "bg-violet-500/10 text-violet-700",
  Retention: "bg-muted text-muted-foreground",
};

const statusDot: Record<PatientStatus, string> = {
  New: "bg-emerald-500",
  Pending: "bg-amber-500",
  "In Review": "bg-sky-500",
  Analyzed: "bg-primary",
  Active: "bg-violet-500",
  Retention: "bg-muted-foreground",
};

function PatientsPage() {
  const [q, setQ] = useState("");
  const [patients, setPatients] = useState<Patient[]>(initialPatients);

  const filtered = patients.filter(
    (p) =>
      p.name.toLowerCase().includes(q.toLowerCase()) ||
      p.id.toLowerCase().includes(q.toLowerCase()),
  );

  const allStatuses: PatientStatus[] = ["New", "Pending", "In Review", "Analyzed", "Active", "Retention"];

  function updateStatus(id: string, status: PatientStatus) {
    setPatients((prev) => prev.map((p) => (p.id === id ? { ...p, status } : p)));
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Patients</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Manage your patient roster and treatment plans.
          </p>
        </div>
        <Button className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95">
          <Plus className="mr-2 h-4 w-4" /> Add patient
        </Button>
      </div>

      <Card className="border-border/60 shadow-[var(--shadow-card)]">
        <CardContent className="p-4 md:p-6">
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative flex-1 min-w-[240px]">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search by name or ID..."
                className="h-10 pl-9"
              />
            </div>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" /> Filter
            </Button>
          </div>

          <div className="mt-4 overflow-hidden rounded-lg border border-border/60">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/40 hover:bg-muted/40">
                  <TableHead>Patient</TableHead>
                  <TableHead>ID</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Treatment</TableHead>
                  <TableHead>Last visit</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filtered.map((p) => (
                  <TableRow key={p.id} className="group">
                    <TableCell>
                      <div className="flex items-center gap-3">
                        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                          {p.name.split(" ").map((n) => n[0]).join("")}
                        </div>
                        <span className="font-medium">{p.name}</span>
                      </div>
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">{p.id}</TableCell>
                    <TableCell>{p.age}</TableCell>
                    <TableCell className="text-sm">{p.treatment}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">{p.lastVisit}</TableCell>
                    <TableCell>
                      <Select
                        value={p.status}
                        onValueChange={(v) => updateStatus(p.id, v as PatientStatus)}
                      >
                        <SelectTrigger className={`h-8 w-[140px] border-0 text-xs font-medium ${statusTone[p.status]} bg-transparent hover:opacity-80`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {allStatuses.map((s) => (
                            <SelectItem key={s} value={s}>
                              <div className="flex items-center gap-2">
                                <span className={`inline-block h-2 w-2 rounded-full ${statusDot[s]}`} />
                                {s}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        asChild
                        variant="ghost"
                        size="sm"
                        className="opacity-0 group-hover:opacity-100"
                      >
                        <Link to="/analysis">View case</Link>
                      </Button>
                      <Button variant="ghost" size="icon">
                        <MoreHorizontal className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
                {filtered.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={7} className="py-12 text-center text-sm text-muted-foreground">
                      No patients found
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
