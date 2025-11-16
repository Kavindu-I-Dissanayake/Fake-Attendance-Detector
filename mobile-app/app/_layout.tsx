import { Slot } from "expo-router";
import { AttendanceProvider } from "../context/AttendanceContext";

export default function RootLayout() {
  return (
    <AttendanceProvider>
      <Slot />
    </AttendanceProvider>
  );
}
