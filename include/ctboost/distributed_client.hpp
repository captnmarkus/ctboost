#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ctboost/distributed_root.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace ctboost {

struct DistributedTcpRoot {
  bool is_tcp{false};
  std::string host;
  std::uint16_t port{0};
  std::string auth_token;
};

inline constexpr std::size_t kDistributedMaxHeaderBytes = 16U * 1024U;
inline constexpr std::size_t kDistributedMaxKeyBytes = 4U * 1024U;
inline constexpr std::size_t kDistributedMaxPayloadBytes = 1024U * 1024U * 1024U;
inline constexpr int kDistributedMaxWorldSize = 65536;

inline bool IsDistributedAuthToken(const std::string& value) {
  if (value.size() != 64U) {
    return false;
  }
  for (const unsigned char character : value) {
    if (std::isxdigit(character) == 0) {
      return false;
    }
  }
  return true;
}

inline DistributedTcpRoot ParseDistributedTcpRoot(const std::string& root) {
  if (!DistributedRootUsesTcp(root)) {
    return DistributedTcpRoot{};
  }
  const std::string endpoint_with_path = root.substr(6);
  const std::size_t slash = endpoint_with_path.find('/');
  const std::string endpoint =
      slash == std::string::npos ? endpoint_with_path : endpoint_with_path.substr(0, slash);
  const std::size_t colon = endpoint.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
    throw std::invalid_argument("distributed tcp root must be formatted like tcp://host:port");
  }
  const std::string raw_port = endpoint.substr(colon + 1);
  if (raw_port.empty() || raw_port.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("distributed tcp port must be an integer");
  }
  std::size_t port_characters = 0U;
  int parsed_port = 0;
  try {
    parsed_port = std::stoi(raw_port, &port_characters);
  } catch (const std::exception&) {
    throw std::invalid_argument("distributed tcp port must be an integer");
  }
  if (port_characters != raw_port.size()) {
    throw std::invalid_argument("distributed tcp port must be an integer");
  }
  if (parsed_port <= 0 || parsed_port > 65535) {
    throw std::invalid_argument("distributed tcp port must be in [1, 65535]");
  }
  const std::string host = endpoint.substr(0, colon);
  if (host == "0.0.0.0" || host == "::") {
    throw std::invalid_argument(
        "distributed tcp root requires a concrete host, not a wildcard address");
  }
  std::string auth_token;
  if (slash != std::string::npos) {
    std::string path = endpoint_with_path.substr(slash + 1U);
    while (!path.empty() && path.back() == '/') {
      path.pop_back();
    }
    if (path.rfind("auth/", 0) == 0) {
      path = path.substr(5);
    }
    if (!IsDistributedAuthToken(path)) {
      throw std::invalid_argument(
          "distributed tcp auth token must contain exactly 64 hexadecimal characters");
    }
    auth_token = std::move(path);
  }
  return DistributedTcpRoot{
      true, host, static_cast<std::uint16_t>(parsed_port), std::move(auth_token)};
}

namespace distributed_client_detail {

#ifdef _WIN32
using SocketHandle = SOCKET;
constexpr SocketHandle kInvalidSocket = INVALID_SOCKET;
#else
using SocketHandle = int;
constexpr SocketHandle kInvalidSocket = -1;
#endif

struct SocketGuard {
  SocketHandle socket{kInvalidSocket};
  ~SocketGuard() {
    if (socket != kInvalidSocket) {
#ifdef _WIN32
      closesocket(socket);
#else
      close(socket);
#endif
    }
  }
};

#ifdef _WIN32
inline void EnsureSocketLibraryInitialized() {
  static bool initialized = []() {
    WSADATA data;
    const int result = WSAStartup(MAKEWORD(2, 2), &data);
    if (result != 0) {
      throw std::runtime_error("WSAStartup failed for distributed tcp client");
    }
    return true;
  }();
  (void)initialized;
}
#else
inline void EnsureSocketLibraryInitialized() {}
#endif

inline void SetSocketTimeout(SocketHandle socket, double timeout_seconds) {
  const double bounded_seconds = timeout_seconds > 0.0 ? timeout_seconds : 0.001;
#ifdef _WIN32
  const DWORD timeout_ms = static_cast<DWORD>(
      std::min(bounded_seconds * 1000.0, static_cast<double>(0xFFFFFFFEUL)));
  if (setsockopt(socket,
                 SOL_SOCKET,
                 SO_RCVTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms)) != 0 ||
      setsockopt(socket,
                 SOL_SOCKET,
                 SO_SNDTIMEO,
                 reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms)) != 0) {
    throw std::runtime_error("failed to configure distributed tcp socket timeout");
  }
#else
  timeval timeout{};
  timeout.tv_sec = static_cast<time_t>(bounded_seconds);
  timeout.tv_usec = static_cast<suseconds_t>(
      (bounded_seconds - static_cast<double>(timeout.tv_sec)) * 1000000.0);
  if (setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
      setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
    throw std::runtime_error("failed to configure distributed tcp socket timeout");
  }
#endif
}

inline void SendAll(SocketHandle socket, const char* data, std::size_t size) {
  std::size_t sent = 0;
  while (sent < size) {
#ifdef _WIN32
    const int result =
        send(socket, data + static_cast<std::ptrdiff_t>(sent), static_cast<int>(size - sent), 0);
#else
    const ssize_t result =
        send(socket, data + static_cast<std::ptrdiff_t>(sent), size - sent, 0);
#endif
    if (result <= 0) {
      throw std::runtime_error("distributed tcp send failed");
    }
    sent += static_cast<std::size_t>(result);
  }
}

inline void ReceiveAll(SocketHandle socket, char* data, std::size_t size) {
  std::size_t received = 0;
  while (received < size) {
#ifdef _WIN32
    const int result = recv(
        socket, data + static_cast<std::ptrdiff_t>(received), static_cast<int>(size - received), 0);
#else
    const ssize_t result =
        recv(socket, data + static_cast<std::ptrdiff_t>(received), size - received, 0);
#endif
    if (result <= 0) {
      throw std::runtime_error("distributed tcp receive failed");
    }
    received += static_cast<std::size_t>(result);
  }
}

inline std::string ReceiveLine(SocketHandle socket,
                               std::size_t max_size = kDistributedMaxHeaderBytes) {
  std::string line;
  while (true) {
    if (line.size() >= max_size) {
      throw std::runtime_error("distributed protocol line exceeds the allowed size");
    }
    char value = '\0';
    ReceiveAll(socket, &value, 1);
    if (value == '\n') {
      return line;
    }
    line.push_back(value);
  }
}

inline bool IsAllowedDistributedOperation(const std::string& op) {
  return op == "allgather" || op == "barrier" || op == "broadcast" ||
         op == "gpu_snapshot_reduce" || op == "node_hist_reduce" || op == "ping" ||
         op == "schema_collect";
}

inline std::vector<std::string> SplitProtocolLine(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t tab = line.find('\t', begin);
    if (tab == std::string::npos) {
      fields.push_back(line.substr(begin));
      return fields;
    }
    fields.push_back(line.substr(begin, tab - begin));
    begin = tab + 1U;
  }
}

inline SocketHandle ConnectSocketWithRetry(const DistributedTcpRoot& root, double timeout_seconds) {
  EnsureSocketLibraryInitialized();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_seconds);

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  const std::string port = std::to_string(root.port);

  while (true) {
    addrinfo* results = nullptr;
    const int resolve_result = getaddrinfo(root.host.c_str(), port.c_str(), &hints, &results);
    if (resolve_result != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("distributed tcp getaddrinfo failed");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      continue;
    }

    SocketHandle connected = kInvalidSocket;
    for (addrinfo* current = results; current != nullptr; current = current->ai_next) {
      SocketHandle candidate =
          socket(current->ai_family, current->ai_socktype, current->ai_protocol);
      if (candidate == kInvalidSocket) {
        continue;
      }
      const int connect_result = connect(candidate, current->ai_addr, static_cast<int>(current->ai_addrlen));
      if (connect_result == 0) {
        connected = candidate;
        break;
      }
#ifdef _WIN32
      closesocket(candidate);
#else
      close(candidate);
#endif
    }
    freeaddrinfo(results);

    if (connected != kInvalidSocket) {
      return connected;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("timed out connecting to distributed tcp coordinator");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

}  // namespace distributed_client_detail

inline std::vector<std::uint8_t> DistributedTcpRequest(const std::string& root,
                                                       double timeout_seconds,
                                                       const std::string& op,
                                                       const std::string& key,
                                                       int rank,
                                                       int world_size,
                                                       const std::vector<std::uint8_t>& payload) {
  const DistributedTcpRoot parsed = ParseDistributedTcpRoot(root);
  if (!parsed.is_tcp) {
    throw std::invalid_argument("distributed tcp request requires a tcp://host:port root");
  }
  if (parsed.auth_token.empty()) {
    throw std::invalid_argument(
        "distributed tcp requests require an authenticated root ending in "
        "'/auth/<64-hex-token>'");
  }
  if (!distributed_client_detail::IsAllowedDistributedOperation(op)) {
    throw std::invalid_argument("unsupported distributed coordinator operation");
  }
  if (key.empty() || key.size() > kDistributedMaxKeyBytes ||
      key.find('\t') != std::string::npos || key.find('\n') != std::string::npos) {
    throw std::invalid_argument(
        "distributed coordinator key is empty or exceeds protocol limits");
  }
  if (world_size <= 0 || world_size > kDistributedMaxWorldSize) {
    throw std::invalid_argument("distributed world_size exceeds protocol limits");
  }
  if (rank < 0 || rank >= world_size) {
    throw std::invalid_argument("distributed rank must be in [0, world_size)");
  }
  if (payload.size() > kDistributedMaxPayloadBytes) {
    throw std::invalid_argument("distributed request payload exceeds the allowed size");
  }
  if ((op == "ping" || op == "barrier") && !payload.empty()) {
    throw std::invalid_argument("distributed ping/barrier requests must not contain a payload");
  }
  if (op == "broadcast" && rank != 0 && !payload.empty()) {
    throw std::invalid_argument(
        "non-root distributed broadcast requests must not contain a payload");
  }

  distributed_client_detail::SocketGuard socket{
      distributed_client_detail::ConnectSocketWithRetry(parsed, timeout_seconds)};
  distributed_client_detail::SetSocketTimeout(socket.socket, timeout_seconds);
  const std::string header = "CTB1\t" + parsed.auth_token + "\t" + op + "\t" + key + "\t" +
                             std::to_string(rank) + "\t" + std::to_string(world_size) + "\t" +
                             std::to_string(payload.size()) + "\n";
  if (header.size() > kDistributedMaxHeaderBytes) {
    throw std::invalid_argument("distributed request header exceeds the allowed size");
  }
  distributed_client_detail::SendAll(socket.socket, header.data(), header.size());
  if (!payload.empty()) {
    distributed_client_detail::SendAll(
        socket.socket, reinterpret_cast<const char*>(payload.data()), payload.size());
  }
  const std::string response_line = distributed_client_detail::ReceiveLine(socket.socket);
  const std::vector<std::string> response_fields =
      distributed_client_detail::SplitProtocolLine(response_line);
  if (response_fields.size() != 3U || response_fields[0] != "CTB1" ||
      (response_fields[1] != "ok" && response_fields[1] != "error")) {
    throw std::runtime_error("invalid authenticated distributed coordinator response");
  }
  std::size_t response_size = 0U;
  if (response_fields[2].empty() ||
      response_fields[2].find_first_not_of("0123456789") != std::string::npos) {
    throw std::runtime_error("invalid distributed coordinator response size");
  }
  try {
    const unsigned long long parsed_response_size = std::stoull(response_fields[2]);
    if (parsed_response_size >
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
      throw std::out_of_range("distributed coordinator response size is too large");
    }
    response_size = static_cast<std::size_t>(parsed_response_size);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid distributed coordinator response size");
  }
  const std::size_t response_limit =
      response_fields[1] == "ok" ? kDistributedMaxPayloadBytes : 4U * 1024U;
  if (response_size > response_limit) {
    throw std::runtime_error("distributed coordinator response exceeds the allowed size");
  }
  std::vector<std::uint8_t> response(response_size, 0U);
  if (response_size != 0U) {
    distributed_client_detail::ReceiveAll(
        socket.socket, reinterpret_cast<char*>(response.data()), response.size());
  }
  if (response_fields[1] != "ok") {
    throw std::runtime_error(
        response.empty()
            ? "distributed coordinator rejected the request"
            : std::string(reinterpret_cast<const char*>(response.data()), response.size()));
  }
  return response;
}

}  // namespace ctboost
