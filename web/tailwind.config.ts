import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0B0E14",
          900: "#0E121A",
          850: "#121722",
          800: "#161C28",
          700: "#1A2130",
          600: "#252D3C",
          500: "#5C6677",
          400: "#778295",
          300: "#9AA6B8",
          200: "#C2CBD8",
          100: "#E7ECF3",
        },
        signal: {
          DEFAULT: "#5EE6C7",
          dim: "#2E6E61",
        },
        bear: "#F26D5B",
        warm: "#E8A13A",
        neutral: "#8A93A6",
        cool: "#3FB6A8",
        bull: "#54C98C",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      letterSpacing: {
        label: "0.18em",
      },
      borderRadius: {
        panel: "14px",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.5s cubic-bezier(0.22, 1, 0.36, 1) both",
        "pulse-soft": "pulse-soft 2.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
