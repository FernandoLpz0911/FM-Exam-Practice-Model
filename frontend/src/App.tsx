import { useState } from 'react'
import StudySession from './components/StudySession'
import Dashboard from './components/Dashboard'
import MockExam from './components/MockExam'
import './App.css'

type Tab = 'study' | 'dashboard' | 'mock-exam'

export default function App() {
  const [tab, setTab] = useState<Tab>('study')

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-title">Exam&nbsp;P&nbsp;Engine</span>
        <nav className="app-nav">
          <button
            className={tab === 'study' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setTab('study')}
          >
            Study
          </button>
          <button
            className={tab === 'dashboard' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setTab('dashboard')}
          >
            Dashboard
          </button>
          <button
            className={tab === 'mock-exam' ? 'nav-btn active' : 'nav-btn'}
            onClick={() => setTab('mock-exam')}
          >
            Mock Exam
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === 'study' && <StudySession />}
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'mock-exam' && <MockExam />}
      </main>
    </div>
  )
}
