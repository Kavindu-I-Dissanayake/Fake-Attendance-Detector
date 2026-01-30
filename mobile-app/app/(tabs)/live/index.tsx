// app/(tabs)/live/index.tsx
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import { useRouter } from "expo-router";

export default function LiveFaceMenu() {
  const router = useRouter();

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Live Face Attendance</Text>

      <Text style={styles.sub}>
        Enroll students or take attendance using face recognition.
      </Text>

      <TouchableOpacity
        style={styles.button}
        onPress={() => router.push("/live/enroll")}
        activeOpacity={0.85}
      >
        <Text style={styles.buttonText}>Enroll Student</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.button}
        onPress={() => router.push("/live/attendance")}
        activeOpacity={0.85}
      >
        <Text style={styles.buttonText}>Take Attendance</Text>
      </TouchableOpacity>

      <Text style={styles.note}>
        Class: <Text style={styles.bold}>Group-Project</Text>
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111", // ✅ same as Home
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  title: {
    color: "#fff",
    fontSize: 26, // ✅ same as Home
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 10,
  },
  sub: {
    color: "#aaa", // ✅ same as Home
    fontSize: 16,
    textAlign: "center",
    marginBottom: 24,
  },
  button: {
    width: "100%",
    maxWidth: 320,
    paddingVertical: 14,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#333",
    backgroundColor: "#000",
    marginBottom: 14,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  note: {
    marginTop: 10,
    fontSize: 12,
    color: "#888",
    textAlign: "center",
  },
  bold: {
    color: "#fff",
    fontWeight: "bold",
  },
});
