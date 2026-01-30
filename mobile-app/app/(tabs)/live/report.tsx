// app/(tabs)/live/report.tsx
import React, { useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  ScrollView,
} from "react-native";
import { useLocalSearchParams } from "expo-router";
import * as Linking from "expo-linking";
import { BASE_URL } from "../../../config";

type Student = { studentId: string; name: string };
type Report = {
  sessionId: string;
  className: string;
  presentCount: number;
  absentCount: number;
  present: Student[];
  absent: Student[];
};

export default function ReportScreen() {
  const params = useLocalSearchParams<{ sessionId?: string | string[] }>();

  // ✅ safely get sessionId as a string
  const sessionId = useMemo(() => {
    const raw = params.sessionId;
    if (Array.isArray(raw)) return raw[0];
    return raw;
  }, [params.sessionId]);

  const reportUrl = sessionId ? `${BASE_URL}/face/session/report/${sessionId}` : "";
  const pdfUrl = sessionId ? `${BASE_URL}/face/session/report/${sessionId}/pdf` : "";

  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<Report | null>(null);
  const [errorText, setErrorText] = useState<string>("");

  const fetchReport = async () => {
    if (!sessionId) {
      Alert.alert("No sessionId", "Open Report from Attendance after starting a session.");
      return;
    }

    setLoading(true);
    setErrorText("");
    setReport(null);

    try {
      const res = await fetch(reportUrl);

      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        const errText = errJson?.detail
          ? String(errJson.detail)
          : await res.text().catch(() => "Failed to fetch report");
        throw new Error(errText || "Failed to fetch report");
      }

      const data = (await res.json()) as Report;
      setReport(data);
    } catch (e: any) {
      const msg = e?.message || "Unknown error";
      setErrorText(msg);
      Alert.alert("Report Error", msg);
    } finally {
      setLoading(false);
    }
  };

  const openPdf = async () => {
    if (!sessionId) {
      Alert.alert("No sessionId", "Open Report from Attendance after starting a session.");
      return;
    }

    try {
      const ok = await Linking.canOpenURL(pdfUrl);
      if (!ok) {
        Alert.alert("Cannot open", "Your device can't open this PDF link.");
        return;
      }
      await Linking.openURL(pdfUrl);
    } catch {
      Alert.alert("Error", "Failed to open PDF link.");
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Attendance Report</Text>

      {/* Debug info (helps if something breaks) */}
      <View style={styles.debugBox}>
        <Text style={styles.debugText}>
          sessionId: <Text style={styles.bold}>{sessionId || "MISSING"}</Text>
        </Text>
        <Text style={styles.debugText}>
          JSON URL: <Text style={styles.dim}>{reportUrl || "N/A"}</Text>
        </Text>
        <Text style={styles.debugText}>
          PDF URL: <Text style={styles.dim}>{pdfUrl || "N/A"}</Text>
        </Text>
      </View>

      {/* Buttons */}
      <TouchableOpacity
        style={[styles.button, { opacity: loading ? 0.7 : 1 }]}
        onPress={fetchReport}
        disabled={loading}
      >
        {loading ? (
          <ActivityIndicator />
        ) : (
          <Text style={styles.buttonText}>Load Report</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.secondaryBtn, { opacity: sessionId ? 1 : 0.5 }]}
        onPress={openPdf}
        disabled={!sessionId}
      >
        <Text style={styles.secondaryText}>Download PDF</Text>
      </TouchableOpacity>

      {!!errorText && <Text style={styles.error}>{errorText}</Text>}

      {!report ? (
        <Text style={styles.helper}>
          Tap “Load Report” to view Present & Absent lists. Tap “Download PDF” to open the PDF.
        </Text>
      ) : (
        <View style={{ width: "100%", maxWidth: 420 }}>
          <Text style={styles.sectionTitle}>Present ({report.presentCount})</Text>

          <View style={styles.listBox}>
            {report.present.length === 0 ? (
              <Text style={styles.itemDim}>No one marked present yet.</Text>
            ) : (
              report.present.map((s) => (
                <Text key={s.studentId} style={styles.item}>
                  • {s.studentId} — {s.name}
                </Text>
              ))
            )}
          </View>

          <Text style={[styles.sectionTitle, { marginTop: 16 }]}>
            Absent ({report.absentCount})
          </Text>

          <View style={styles.listBox}>
            {report.absent.length === 0 ? (
              <Text style={styles.itemDim}>No absentees.</Text>
            ) : (
              report.absent.map((s) => (
                <Text key={s.studentId} style={styles.item}>
                  • {s.studentId} — {s.name}
                </Text>
              ))
            )}
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: "#111",
    alignItems: "center",
    padding: 20,
    paddingTop: 30,
  },
  title: {
    color: "#fff",
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
  },
  bold: { color: "#fff", fontWeight: "bold" },
  dim: { color: "#aaa" },

  debugBox: {
    width: "100%",
    maxWidth: 420,
    borderWidth: 1,
    borderColor: "#333",
    borderRadius: 12,
    padding: 12,
    backgroundColor: "#000",
    marginTop: 14,
    marginBottom: 12,
  },
  debugText: { color: "#ddd", fontSize: 12, marginBottom: 6 },

  button: {
    width: "100%",
    maxWidth: 320,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
    backgroundColor: "#000",
    marginBottom: 10,
    alignItems: "center",
  },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },

  secondaryBtn: {
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 10,
  },
  secondaryText: {
    color: "#aaa",
    textDecorationLine: "underline",
    fontSize: 14,
  },

  helper: { color: "#888", fontSize: 12, textAlign: "center", marginTop: 8 },
  error: { color: "#ff7b7b", fontSize: 13, marginTop: 6, textAlign: "center" },

  sectionTitle: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "700",
    marginBottom: 8,
  },
  listBox: {
    borderWidth: 1,
    borderColor: "#333",
    backgroundColor: "#000",
    borderRadius: 12,
    padding: 12,
  },
  item: { color: "#ddd", fontSize: 14, marginBottom: 6 },
  itemDim: { color: "#888", fontSize: 14 },
});
