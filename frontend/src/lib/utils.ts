import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number, decimals: number = 2): string {
  if (num === null || num === undefined || isNaN(num)) return "N/A"
  return new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: decimals,
    minimumFractionDigits: decimals > 0 ? decimals : 0
  }).format(num)
}

export function formatYield(val: number): string {
  return `${formatNumber(val, 1)} kg/ha`
}

export function formatProduction(val: number): string {
  return `${formatNumber(val, 2)} '000 t`
}

export function formatArea(val: number): string {
  return `${formatNumber(val, 2)} '000 ha`
}

export function formatPercent(val: number): string {
  return `${formatNumber(val, 1)}%`
}
