import { EstimateNav } from "@/components/estimate/EstimateNav";
import { Separator } from "@/components/ui/separator"

export default function EstimateLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <div className="flex flex-col gap-4 p-8">
            <EstimateNav />
            <Separator />
            {children}
        </div>
    );
  }