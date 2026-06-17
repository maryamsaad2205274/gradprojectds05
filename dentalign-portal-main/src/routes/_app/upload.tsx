import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UploadCloud, ImagePlus, X, ArrowRight, Camera, Lock, StickyNote } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/_app/upload")({
  head: () => ({ meta: [{ title: "Upload Case — DentAlign" }] }),
  component: UploadPage,
});

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
            <p className="mt-3 text-sm font-medium">Click or drag photo here</p>
            <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
          </>
        )}
        <input
          ref={ref}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] ?? undefined)}
        />
      </div>
    </div>
  );
}

function UploadPage() {
  const navigate = useNavigate();
  const [front, setFront] = useState<string | null>(null);
  const [side, setSide] = useState<string | null>(null);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!front || !side) {
      toast.error("Please upload both front and side facial photos.");
      return;
    }
    toast.success("Case uploaded. Running landmark analysis…");
    setTimeout(() => navigate({ to: "/analysis" }), 800);
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Upload new case</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Add patient details and facial photographs to begin cephalometric analysis.
        </p>
      </div>

      <form onSubmit={submit} className="space-y-6">
        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <CardTitle className="text-base">Patient information</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Patient name</Label>
              <Input placeholder="e.g. Sama" required />
            </div>
            <div className="space-y-2">
              <Label>Patient ID</Label>
              <Input placeholder="PT-1043" required />
            </div>
            <div className="space-y-2">
              <Label>Date of birth</Label>
              <Input type="date" required />
            </div>
            <div className="space-y-2">
              <Label>Case type</Label>
              <Select defaultValue="class2">
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="class1">Class I</SelectItem>
                  <SelectItem value="class2">Class II</SelectItem>
                  <SelectItem value="class3">Class III</SelectItem>
                  <SelectItem value="openbite">Open bite</SelectItem>
                  <SelectItem value="crowding">Crowding</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <div>
              <CardTitle className="text-base">Facial photographs</CardTitle>
              <p className="mt-1 text-xs text-muted-foreground">
                JPG or PNG, well-lit and centered. Both required for analysis.
              </p>
            </div>
            <Button type="button" variant="outline" size="sm">
              <Camera className="mr-2 h-4 w-4" /> Capture from device
            </Button>
          </CardHeader>
          <CardContent className="grid gap-6 md:grid-cols-2">
            <PhotoDrop
              label="Frontal view"
              hint="Patient facing camera, lips relaxed"
              value={front}
              onChange={setFront}
            />
            <PhotoDrop
              label="Lateral (side) view"
              hint="True profile, Frankfort plane horizontal"
              value={side}
              onChange={setSide}
            />
          </CardContent>
        </Card>

        <Card className="border-border/60 shadow-[var(--shadow-card)]">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <StickyNote className="h-4 w-4 text-primary" />
                <CardTitle className="text-base">Private notes</CardTitle>
              </div>
              <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                <Lock className="h-3 w-3" /> Visible only to you
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Personal reminders, observations, or follow-up items. Not included in patient reports.
            </p>
          </CardHeader>
          <CardContent>
            <Textarea
              placeholder="e.g. Bassant reports clicking on right TMJ. Re-check anchorage at next visit…"
              className="min-h-[120px] resize-y"
            />
          </CardContent>
        </Card>

        <div className="flex items-center justify-between rounded-xl border border-border/60 bg-card p-4 shadow-[var(--shadow-card)]">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <UploadCloud className="h-5 w-5" />
            </div>
            <div>
              <p className="text-sm font-medium">Ready to analyze</p>
              <p className="text-xs text-muted-foreground">
                Photos are encrypted in transit and at rest.
              </p>
            </div>
          </div>
          <Button
            type="submit"
            className="bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
          >
            Run analysis <ArrowRight className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}
