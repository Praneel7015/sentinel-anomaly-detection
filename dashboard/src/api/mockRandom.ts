/** Small deterministic PRNG helpers so the mock dataset is byte-identical across reloads. */

export type Rng = () => number

/** mulberry32 — fast, seedable, good enough for demo data. */
export function makeRng(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function pick<T>(rng: Rng, items: readonly T[]): T {
  return items[Math.floor(rng() * items.length) % items.length]
}

export function pickMany<T>(rng: Rng, items: readonly T[], count: number): T[] {
  const pool = [...items]
  const out: T[] = []
  for (let i = 0; i < count && pool.length > 0; i++) {
    out.push(pool.splice(Math.floor(rng() * pool.length), 1)[0])
  }
  return out
}

export function randInt(rng: Rng, min: number, max: number): number {
  return min + Math.floor(rng() * (max - min + 1))
}

export function randFloat(rng: Rng, min: number, max: number): number {
  return min + rng() * (max - min)
}

/** Box-Muller, clamped to +/-3 sigma to avoid silly outliers. */
export function gauss(rng: Rng, mean: number, sd: number): number {
  const u = Math.max(rng(), 1e-9)
  const v = rng()
  const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  return mean + sd * clamp(z, -3, 3)
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

export function chance(rng: Rng, probability: number): boolean {
  return rng() < probability
}

export function round(value: number, digits = 2): number {
  const f = 10 ** digits
  return Math.round(value * f) / f
}
