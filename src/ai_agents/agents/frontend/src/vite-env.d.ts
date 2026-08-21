/// <reference types="vite/client" />

interface DesktopDirectoryPickerOptions {
  title?: string;
  defaultPath?: string;
}

interface DesktopBridge {
  platform: string;

  selectDirectory?: (
    options?: DesktopDirectoryPickerOptions,
  ) => Promise<string | null>;
}

interface Window {
  desktop?: DesktopBridge;
}
