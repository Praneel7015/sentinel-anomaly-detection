import { useCallback, useEffect, useState } from 'react'
import { TopBar } from './components/layout/TopBar'
import { EntityView } from './components/entity/EntityView'
import { OpsView } from './components/ops/OpsView'
import { TriageView } from './components/triage/TriageView'
import { StreamProvider } from './hooks/useStream'
import type { ViewId } from './views'

export type Theme = 'light' | 'dark'

function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem('sentinel-theme') as Theme | null
    if (stored === 'light' || stored === 'dark') return stored
  } catch {}
  return 'light'
}

export default function App() {
  const [view, setView] = useState<ViewId>('triage')
  const [entityId, setEntityId] = useState<string | null>(null)
  const [focusEventId, setFocusEventId] = useState<string | null>(null)
  const [budgetPct, setBudgetPct] = useState(1)
  const [theme, setTheme] = useState<Theme>(getInitialTheme)

  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    try { localStorage.setItem('sentinel-theme', theme) } catch {}
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'light' ? 'dark' : 'light'))
  }, [])

  const openEntity = useCallback((id: string) => {
    setEntityId(id)
    setView('entity')
  }, [])

  const openAlert = useCallback((eventId: string) => {
    setFocusEventId(eventId)
    setView('triage')
  }, [])

  return (
    <StreamProvider>
      <div className="flex h-full min-h-0 flex-col bg-surface-0">
        <TopBar view={view} onNavigate={setView} theme={theme} onToggleTheme={toggleTheme} />
        <main className="min-h-0 flex-1 overflow-hidden">
          {view === 'triage' && (
            <TriageView
              budgetPct={budgetPct}
              onBudgetChange={setBudgetPct}
              onOpenEntity={openEntity}
              focusEventId={focusEventId}
              onFocusConsumed={() => setFocusEventId(null)}
            />
          )}
          {view === 'entity' && (
            <EntityView entityId={entityId} onSelectEntity={setEntityId} onOpenAlert={openAlert} />
          )}
          {view === 'ops' && <OpsView budgetPct={budgetPct} onBudgetChange={setBudgetPct} />}
        </main>
      </div>
    </StreamProvider>
  )
}
