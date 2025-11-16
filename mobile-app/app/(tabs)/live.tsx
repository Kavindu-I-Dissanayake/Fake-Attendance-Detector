import { View, Text, StyleSheet } from "react-native";

export default function LiveScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Live Face Recognition</Text>
      <Text style={styles.sub}>Feature coming soon...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#111", justifyContent: "center", alignItems: "center" },
  title: { color: "#fff", fontSize: 26, fontWeight: "bold" },
  sub: { color: "#777", marginTop: 10 },
});
