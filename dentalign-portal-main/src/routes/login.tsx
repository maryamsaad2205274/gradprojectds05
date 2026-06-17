import { createFileRoute, useNavigate, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Sparkles, ShieldCheck, Activity, Lock } from "lucide-react";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [
      { title: "Sign in — DentAlign" },
      { name: "description", content: "Secure clinician login to DentAlign orthodontic platform." },
    ],
  }),
  component: LoginPage,
});

function LoginPage() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => navigate({ to: "/dashboard" }), 600);
  };

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      {/* Left brand panel */}
      <div className="relative hidden overflow-hidden bg-[image:var(--gradient-primary)] p-12 text-primary-foreground lg:flex lg:flex-col lg:justify-between">
        <div className="absolute -right-32 -top-32 h-96 w-96 rounded-full bg-white/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-20 h-96 w-96 rounded-full bg-white/10 blur-3xl" />
        <div className="relative flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/15 backdrop-blur">
            <Sparkles className="h-6 w-6" />
          </div>
          <div>
            <p className="text-xl font-bold tracking-tight">DentAlign</p>
            <p className="text-xs uppercase tracking-widest text-white/70">Orthodontic Suite</p>
          </div>
        </div>

        <div className="relative space-y-6">
          <h1 className="text-4xl font-bold leading-tight tracking-tight">
            Precision orthodontic analysis,<br />designed for clinicians.
          </h1>
          <p className="max-w-md text-white/80">
            Upload facial and intraoral photographs, generate AI-assisted landmark
            analyses, and share secure progress reports with your patients.
          </p>
          <div className="grid gap-3 pt-4">
            {[
              { icon: ShieldCheck, t: "HIPAA-aligned secure storage" },
              { icon: Activity, t: "Automated cephalometric landmark detection" },
              { icon: Lock, t: "Encrypted patient portal access" },
            ].map(({ icon: Icon, t }) => (
              <div key={t} className="flex items-center gap-3 text-sm text-white/90">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/15">
                  <Icon className="h-4 w-4" />
                </div>
                {t}
              </div>
            ))}
          </div>
        </div>

        <p className="relative text-xs text-white/60">© 2026 DentAlign Health Systems</p>
      </div>

      {/* Right form */}
      <div className="flex items-center justify-center bg-background p-6">
        <Card className="w-full max-w-md border-border/60 shadow-[var(--shadow-card)]">
          <CardContent className="p-8">
            <div className="mb-8 flex items-center gap-2 lg:hidden">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[image:var(--gradient-primary)] text-primary-foreground">
                <Sparkles className="h-5 w-5" />
              </div>
              <span className="text-lg font-bold">DentAlign</span>
            </div>

            <h2 className="text-2xl font-semibold tracking-tight">Welcome back, Doctor</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Sign in to access your clinical dashboard.
            </p>

            <form onSubmit={submit} className="mt-8 space-y-5">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="sama@clinic.com"
                  defaultValue="dr.smith@dentalign.health"
                  required
                  className="h-11"
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                  <a href="#" className="text-xs font-medium text-primary hover:underline">
                    Forgot?
                  </a>
                </div>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  defaultValue="demopassword"
                  required
                  className="h-11"
                />
              </div>

              <Button
                type="submit"
                disabled={loading}
                className="h-11 w-full bg-[image:var(--gradient-primary)] shadow-[var(--shadow-elegant)] hover:opacity-95"
              >
                {loading ? "Signing in…" : "Sign in"}
              </Button>
            </form>

            <div className="mt-8 rounded-lg border border-border/60 bg-muted/40 p-4 text-center text-xs text-muted-foreground">
              Are you a patient?{" "}
              <Link to="/portal" className="font-medium text-primary hover:underline">
                Access patient portal →
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
