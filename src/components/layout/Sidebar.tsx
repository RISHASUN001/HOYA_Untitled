import { Home, Calendar, Users, MessageSquare, Video, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

const navigation = [
  { name: 'Activity', icon: Home },
  { name: 'Chat', icon: MessageSquare },
  { name: 'Teams', icon: Users },
  { name: 'Calendar', icon: Calendar },
  { name: 'Calls', icon: Video },
];

export function Sidebar() {
  return (
    <div className="flex h-full w-[68px] flex-col items-center border-r bg-card px-2 py-4">
      <div className="mb-4">
        <Search className="h-5 w-5 text-muted-foreground" />
      </div>
      {navigation.map((item) => {
        const Icon = item.icon;
        return (
          <Button
            key={item.name}
            variant="ghost"
            size="icon"
            className={cn(
              'group relative my-1 h-12 w-12 rounded-md',
              'hover:bg-accent text-muted-foreground hover:text-accent-foreground'
            )}
          >
            <Icon className="h-5 w-5" />
            <div className="absolute left-full ml-2 hidden rounded-md bg-popover px-2 py-1 text-sm text-popover-foreground group-hover:block">
              {item.name}
            </div>
          </Button>
        );
      })}
    </div>
  );
}