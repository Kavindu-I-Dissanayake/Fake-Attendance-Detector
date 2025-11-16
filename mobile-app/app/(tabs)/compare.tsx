import { View, Text, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useAttendance } from "../../context/AttendanceContext";

export default function CompareScreen() {
  const { headCount, signatureCount, absentees } = useAttendance();

  let statusMessage = "";
  let statusColor = "#444";

  if (headCount == null || signatureCount == null) {
    statusMessage = "Upload video + signature sheet to compare.";
  } else if (headCount < signatureCount) {
    statusMessage = "❌ MISMATCH — Possible Fake Signatures!";
    statusColor = "#CC0000";
  } else if (headCount > signatureCount) {
    statusMessage = "⚠️ MISMATCH — Someone did not sign!";
    statusColor = "#CC8800";
  } else {
    statusMessage = "✅ MATCH — Attendance Verified!";
    statusColor = "#009944";
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView>
        <Text style={styles.title}>Attendance Comparison</Text>

        {/* Counts card */}
        <View style={styles.card}>
          <Text style={styles.label}>Head Count (Video):</Text>
          <Text style={styles.value}>{headCount ?? "--"}</Text>

          <Text style={[styles.label, { marginTop: 18 }]}>
            Signature Count (Sheet):
          </Text>
          <Text style={styles.value}>{signatureCount ?? "--"}</Text>
        </View>

        {/* Status message */}
        <View style={[styles.statusBox, { backgroundColor: statusColor }]}>
          <Text style={styles.statusText}>{statusMessage}</Text>
        </View>

        {/* Absentees List */}
        <Text style={styles.absTitle}>Absentees (Did NOT Sign)</Text>

        {absentees.length === 0 ? (
          <Text style={styles.noAbs}>No absentees detected.</Text>
        ) : (
          <View style={styles.absList}>
            {absentees.map((item, idx) => (
              <View key={idx} style={styles.absItem}>
                <Text style={styles.absName}>
                  {item.serial}. {item.name}
                </Text>
                <Text style={styles.absReg}>{item.reg_no}</Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#1E1E1E",
    padding: 15,
  },
  title: {
    color: "white",
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 15,
  },
  card: {
    backgroundColor: "#2A2A2A",
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
  },
  label: {
    color: "#AAAAAA",
    fontSize: 16,
  },
  value: {
    color: "white",
    fontSize: 28,
    fontWeight: "bold",
    marginTop: 4,
  },
  statusBox: {
    padding: 18,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 25,
  },
  statusText: {
    fontSize: 18,
    fontWeight: "bold",
    color: "white",
  },
  absTitle: {
    fontSize: 20,
    color: "#fff",
    fontWeight: "bold",
    marginBottom: 10,
    marginTop: 5,
  },
  noAbs: {
    color: "#aaa",
    fontSize: 16,
    fontStyle: "italic",
  },
  absList: {
    backgroundColor: "#222",
    borderRadius: 10,
    padding: 12,
  },
  absItem: {
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: "#333",
  },
  absName: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  absReg: {
    color: "#66AFFF",
    fontSize: 14,
  },
});
