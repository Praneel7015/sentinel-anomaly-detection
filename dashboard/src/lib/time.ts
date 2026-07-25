import { format, formatDistanceToNowStrict, isValid, parseISO } from 'date-fns'

export function parse(iso: string): Date {
  const d = parseISO(iso)
  return isValid(d) ? d : new Date(0)
}

/** "4m", "2h", "3d" — compact enough for a dense alert row. */
export function relative(iso: string): string {
  return formatDistanceToNowStrict(parse(iso))
    .replace(' seconds', 's')
    .replace(' second', 's')
    .replace(' minutes', 'm')
    .replace(' minute', 'm')
    .replace(' hours', 'h')
    .replace(' hour', 'h')
    .replace(' days', 'd')
    .replace(' day', 'd')
    .replace(' months', 'mo')
    .replace(' month', 'mo')
    .replace(' years', 'y')
    .replace(' year', 'y')
    .replace(/\s/g, '')
}

export function absolute(iso: string): string {
  return format(parse(iso), 'yyyy-MM-dd HH:mm:ss')
}

export function shortTime(iso: string): string {
  return format(parse(iso), 'HH:mm:ss')
}

export function dayHour(iso: string): string {
  return format(parse(iso), 'MMM d HH:mm')
}
