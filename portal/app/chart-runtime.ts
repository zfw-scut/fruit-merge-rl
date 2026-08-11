import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  ToolboxComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { LabelLayout } from "echarts/features";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  ToolboxComponent,
  TooltipComponent,
  LabelLayout,
  CanvasRenderer,
]);

export { echarts };
