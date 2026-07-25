/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Layered near-black surfaces, darkest at the back.
        surface: {
          0: '#0a0c10',
          1: '#12151c',
          2: '#1a1f29',
          3: '#232936',
          4: '#2d3442',
        },
        edge: {
          DEFAULT: '#232936',
          strong: '#333c4d',
        },
        ink: {
          DEFAULT: '#e6ebf2',
          dim: '#98a3b5',
          faint: '#5d6879',
        },
        accent: {
          DEFAULT: '#22d3ee',
          dim: '#0e7490',
          glow: '#67e8f9',
        },
        amber: {
          DEFAULT: '#f5a524',
        },
        // Deliberate risk ramp - the single source of truth for risk colour.
        risk: {
          low: '#22c55e',
          medium: '#eab308',
          high: '#f97316',
          critical: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      fontSize: {
        '2xs': ['0.6875rem', { lineHeight: '1rem' }],
      },
      keyframes: {
        'pulse-dot': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.3' },
        },
        'slide-in': {
          from: { opacity: '0', transform: 'translateY(-6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'pulse-dot': 'pulse-dot 2s ease-in-out infinite',
        'slide-in': 'slide-in 220ms ease-out',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
}
