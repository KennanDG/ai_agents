import { contextBridge, ipcRenderer } from "electron";

type DesktopDirectoryPickerOptions = {
  title?: string;
  defaultPath?: string;
};

contextBridge.exposeInMainWorld("desktop", {
  platform: process.platform,

  selectDirectory: (
    options?: DesktopDirectoryPickerOptions,
  ): Promise<string | null> =>
    ipcRenderer.invoke("desktop:select-directory", options),
});