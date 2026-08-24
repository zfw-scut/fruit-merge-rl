import type { Metadata } from "next";
import "./globals.css";
import "./workspace-modules.css";

export const metadata: Metadata = {
  title: "Xigua Atlas · 模型知识与训练控制台",
  description: "合成大西瓜强化学习项目的模型证据、文档、实验与工具控制门户。",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
