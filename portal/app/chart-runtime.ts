import { BarChart, ScatterChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { LabelLayout } from "echarts/features";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  ScatterChart,
  GridComponent,
  TooltipComponent,
  LabelLayout,
  CanvasRenderer,
]);

export { echarts };
