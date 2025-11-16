import { View, Text, StyleSheet } from "react-native";

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Welcome to Fake Attendance Detector</Text>
      <Text style={styles.sub}>
        Use tabs below to detect heads, scan signature sheets, and compare.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111",
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  title: {
    color: "#fff",
    fontSize: 26,
    fontWeight: "bold",
    textAlign: "center",
  },
  sub: {
    color: "#aaa",
    fontSize: 16,
    marginTop: 14,
    textAlign: "center",
  },
});
