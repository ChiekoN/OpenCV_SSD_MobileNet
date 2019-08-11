#ifndef _MESSAGEQUEUE_H
#define _MESSAGEQUEUE_H

#include <mutex>
#include <deque>
#include <condition_variable>

template <typename T>
class MessageQueue
{
  public:
    T receive()
    {
        std::unique_lock<std::mutex> ulock(_mutex);
        _cond.wait(ulock, [this]{ return !_messages.empty(); });
        T msg = std::move(_messages.front());
        _messages.pop_front();

        return msg;
    }

    void send(T &&msg)
    {
        std::lock_guard<std::mutex> ulock(_mutex);
        _messages.push_back(std::move(msg));
        _cond.notify_one();
    }

    size_t getSize()
    {
        return _messages.size();
    }
    
    int getTotal()
    {
        // if _total = 0, sending is unfinished
        // if _total > 0, the total number of messages being sent
        std::lock_guard<std::mutex> ulock(_mutex);
        return _total;
    }
    void setTotal(int total)
    {
        std::lock_guard<std::mutex> ulock(_mutex);
        _total = total;
    }

  private:
    std::mutex _mutex;
    std::condition_variable _cond;
    std::deque<T> _messages;
    int _total = 0;
};

#endif