import { BarChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { LabelLayout } from "echarts/features";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  HeatmapChart,
  GridComponent,
  DataZoomComponent,
  LegendComponent,
  ToolboxComponent,
  TooltipComponent,
  VisualMapComponent,
  LabelLayout,
  CanvasRenderer,
]);

export { echarts };
