"use client";

import type { EChartsOption } from "echarts";
import { useEffect, useRef } from "react";

type EChartProps = {
  option: EChartsOption;
  className?: string;
  onClick?: (payload: { name?: string; dataIndex?: number }) => void;
};

export function EChart({ option, className = "", onClick }: EChartProps) {
  const elementRef = useRef<HTMLDivElement>(null);
  const callbackRef = useRef(onClick);
  const optionRef = useRef(option);
  const chartRef = useRef<import("echarts/core").ECharts | null>(null);

  useEffect(() => {
    callbackRef.current = onClick;
  }, [onClick]);

  useEffect(() => {
    let disposed = false;
    let resizeObserver: ResizeObserver | undefined;
    let chart: import("echarts/core").ECharts | undefined;

    void import("./chart-runtime").then(({ echarts }) => {
      if (disposed || !elementRef.current) return;
      chart = echarts.init(elementRef.current, undefined, { renderer: "canvas" });
      chartRef.current = chart;
      chart.setOption(optionRef.current, { notMerge: true });
      chart.on("click", (payload) => callbackRef.current?.(payload));
      resizeObserver = new ResizeObserver(() => chart?.resize());
      resizeObserver.observe(elementRef.current);
    });

    return () => {
      disposed = true;
      resizeObserver?.disconnect();
      chartRef.current = null;
      chart?.dispose();
    };
  }, []);

  useEffect(() => {
    optionRef.current = option;
    chartRef.current?.setOption(option, { notMerge: true });
  }, [option]);

  return <div ref={elementRef} className={`echart ${className}`} role="img" />;
}
