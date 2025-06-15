class Node:
    def __init__(self, data):
        self.data = data  # Data stored in the node
        self.next = None  # Reference to the next node (initially None)


def add_At_Last(head,data):
    """Adds a node to the end of the list."""
    current = head
    new_node = Node(data)
    if head is None:
        return new_node
    else:
        while current.next is not None:  # Traverse to the end node
            current = current.next
        current.next = new_node  # Link the new node

def print_list(head):
    """Prints the list from head to end."""
    current = head
    if not current:
        print("List is empty.")
        return
    else:
        while current is not None:
            print(current.data, end=" ")
            current = current.next

def delete_nth_node(head, n):
    """
    Deletes the nth node (1-based index).
    Raises IndexError if n is out of range or list is empty.
    """
    if not head:
        raise IndexError("Cannot delete from an empty list.")
    if n < 1:
        raise IndexError("Index must be 1 or greater.")
        
    if n == 1:  # Special case: delete head
        head = head.next
        return
    
    current = head
    for i in range(1, n-1):
        if not current.next:
            raise IndexError("Index out of range.")
        current = current.next
    
    if not current.next:
        raise IndexError("Index out of range.")
        
    current.next = current.next.next
    return head      

# Test the implementation
if __name__ == "__main__":
    head = Node(10)
    head.next = Node(20)
    head.next.next = Node(30)
    head.next.next.next = Node(88)
    head.next.next.next.next = Node(550)
    head.next.next.next.next.next = Node(760)

    # Edge case tests
    try:
        delete_nth_node(head,0)  # Invalid index
    except IndexError as e:
        print(f"\nError: {e}")

    # Test 1: Delete middle node (n=3)
    try:
        head = delete_nth_node(head, 3)
        print("\nAfter deleting 3rd node:")
        print_list(head)  # Expected: 1 -> 2 -> 4 -> 5 -> None
    except IndexError as e:
        print(f"Error: {e}")

    try:
        delete_nth_node(head,8)  # Out of range
    except IndexError as e:
        print(f"Error: {e}")


    try:
        delete_nth_node(head,1)  # Empty list
    except IndexError as e:
        print(f"Error: {e}")