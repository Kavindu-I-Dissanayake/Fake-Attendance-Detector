import React, { createContext, useContext, useState } from "react";

const AttendanceContext = createContext<any>(null);

export function AttendanceProvider({ children }: any) {
  const [headCount, setHeadCount] = useState<number | null>(null);
  const [signatureCount, setSignatureCount] = useState<number | null>(null);
  const [absentees, setAbsentees] = useState<any[]>([]);

  return (
    <AttendanceContext.Provider
      value={{
        headCount,
        setHeadCount,
        signatureCount,
        setSignatureCount,
        absentees,
        setAbsentees,
      }}
    >
      {children}
    </AttendanceContext.Provider>
  );
}

export const useAttendance = () => useContext(AttendanceContext);
