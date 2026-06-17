import { Outlet, createFileRoute, Link, useRouterState } from "@tanstack/react-router";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { Bell, Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/_app")({
  component: AppLayout,
});

function AppLayout() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const crumb =
    path.replace("/", "").split("/")[0]?.replace(/^./, (c) => c.toUpperCase()) ||
    "Dashboard";

  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full bg-[image:var(--gradient-soft)]">
        <AppSidebar />
        <div className="flex flex-1 flex-col">
          <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-border/60 bg-background/80 px-4 backdrop-blur-md">
            <SidebarTrigger />
            <div className="hidden text-sm text-muted-foreground md:flex items-center gap-2">
              <Link to="/dashboard" className="hover:text-foreground">DentAlign</Link>
              <span>/</span>
              <span className="font-medium text-foreground">{crumb}</span>
            </div>
            <div className="ml-auto flex items-center gap-3">
              <div className="relative hidden md:block">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  placeholder="Search patients, cases..."
                  className="h-9 w-72 pl-9 bg-muted/50 border-transparent focus-visible:bg-background"
                />
              </div>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-4 w-4" />
                <span className="absolute right-2 top-2 h-2 w-2 rounded-full bg-primary" />
              </Button>
              <Avatar className="h-9 w-9 ring-2 ring-primary/20">
                <AvatarFallback className="bg-[image:var(--gradient-primary)] text-primary-foreground text-sm font-semibold">
                  DR
                </AvatarFallback>
              </Avatar>
            </div>
          </header>
          <main className="flex-1 p-6 md:p-8">
            <Outlet />
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}
